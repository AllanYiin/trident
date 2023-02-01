import itertools
import sys
import uuid
import copy
import warnings
import weakref
from collections import defaultdict, abc,namedtuple
from types import MethodType
from typing import List, Tuple, Optional, Union, Callable, Any, Iterable,Iterator,Mapping, TypeVar,overload,Dict,Set
from functools import partial
import builtins
import numbers
import numpy as np
import gc
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
from trident.backend import common
from trident.backend.common import to_list, addindent, camel2snake, unpack_singleton, enforce_singleton, OrderedDict, get_session, set_session, get_session_value, \
    PrintException, Signature, TensorShape,  get_args_spec,is_instance
from trident.backend.tensorspec import *
from trident.backend import jax_ops as jops
from trident.backend import dtype as Dtype
from trident import context
from trident.data.utils import pickle_it,unpickle
from trident.context import split_path, make_dir_if_need, sanitize_path
from trident.backend.jax_ops import *

ctx = context._context()
_backend = ctx.get_backend()

__all__ = ['get_device', 'set_device', 'Layer', 'Parameter', 'DTYPE_MAPPING']

_int = builtins.int
_float = builtins.float
_bool = builtins.bool

DTYPE_MAPPING = {
    jnp.bool_: Dtype.bool,
    jnp.int8: Dtype.int8,
    jnp.int16: Dtype.int16,
    jnp.int32: Dtype.int32,
    jnp.int64: Dtype.int64,
    jnp.uint8: Dtype.uint8,
    jnp.float16: Dtype.float16,
    jnp.float32: Dtype.float32,
    jnp.float64: Dtype.float64,
    jnp.complex64: Dtype.complex64,
    jnp.complex128: Dtype.complex128,

}


def get_device():
    """get current device

    Returns: device string ('cpu', 'cuda)

    """
    if get_session().device is None:
        set_device(jax.default_backend())
    return get_session().device


def set_device(device='cpu'):
    device = device.lower().replace('cuda', 'gpu').replace('xpu', 'tpu')
    if device == 'gpu' and not is_gpu_available():
        raise ValueError('Gpu is not available...')
    if device == 'tpu' and not is_tpu_available():
        raise ValueError('Tpu is not available...')
    try:
        device_ = device
        if device == ['xpu', 'tpu']:
            import torch_xla.core.xla_model as xm
            device_ = xm.xla_device()
        elif device in ['cuda', 'gpu']:
            device_ = jax.devices('gpu')[0]
        elif device in ['cpu']:
            device_ = jax.devices()[0]
        set_session('device', device_)

        gcitems = gc.get_objects()
        for i in range(len(gcitems)):
            obj = gcitems[i]
            try:
                if is_tensor(obj):
                    jax.device_put(obj, device=device_)
                elif is_instance(obj, 'Layer'):
                    obj.to(device_)
            except Exception:
                pass
    except Exception as e:
        print(e)


class RemovableHandle(object):
    """A handle which provides the capability to remove a hook."""

    next_id = 0

    def __init__(self, hooks_dict):
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self):
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

    def __getstate__(self):
        return self.hooks_dict_ref(), self.id

    def __setstate__(self, state):
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.remove()
#
# class BackwardHook(object):
#     """
#     A wrapper class to implement nn.Module backward hooks.
#     It handles:
#       - Ignoring non-Tensor inputs and replacing them by None before calling the user hook
#       - Generating the proper Node to capture a set of Tensor's gradients
#       - Linking the gradients captures for the outputs with the gradients captured for the input
#       - Calling the user hook once both output and input gradients are available
#     """
#
#     def __init__(self, module, user_hooks):
#         self.user_hooks = user_hooks
#         self.module = module
#
#         self.grad_outputs = None
#         self.n_outputs = -1
#         self.output_tensors_index = None
#         self.n_inputs = -1
#         self.input_tensors_index = None
#
#     def _pack_with_none(self, indices, values, size):
#         res = [None] * size
#         for idx, val in zip(indices, values):
#             res[idx] = val
#
#         return tuple(res)
#
#     def _unpack_none(self, indices, values):
#         res = []
#         for idx in indices:
#             res.append(values[idx])
#
#         return tuple(res)
#
#     def _set_user_hook(self, grad_fn, user_hook):
#         @functools.wraps(user_hook)
#         def hook(grad_input, _):
#             if self.grad_outputs is None:
#                 raise RuntimeError("Module backward hook for grad_input is called before "
#                                    "the grad_output one. This happens because the gradient "
#                                    "in your nn.Module flows to the Module's input without "
#                                    "passing through the Module's output. Make sure that the "
#                                    "output depends on the input and that the loss is computed "
#                                    "based on the output.")
#
#             grad_input = self._pack_with_none(self.input_tensors_index, grad_input, self.n_inputs)
#             res = user_hook(self.module, grad_input, self.grad_outputs)
#             if res is None:
#                 return res
#
#             if len(res) != len(grad_input):
#                 raise RuntimeError("Backward hook returned an invalid number of grad_input, "
#                                    "got {}, but expected {}".format(len(res), len(grad_input)))
#             return self._unpack_none(self.input_tensors_index, res)
#         grad_fn.register_hook(hook)
#
#     def _apply_on_tensors(self, fn, args):
#         # Can be used to apply the given function to the tensors contained in the
#         # args. Will return updated args and the tensors indices
#         tensors_idx = []
#         tensors = []
#
#         requires_grad = False
#         for i, arg in enumerate(args):
#             if is_tensor(arg):
#                 tensors_idx.append(i)
#                 tensors.append(arg)
#                 requires_grad |= arg.requires_grad
#
#         if not requires_grad:
#             return args, None
#
#         new_tensors = torch.nn.modules._functions.BackwardHookFunction.apply(*tensors)
#         if len(new_tensors) == 0:
#             raise RuntimeError("Cannot set Module backward hook for a Module with no input Tensors.")
#         grad_fn = new_tensors[0].grad_fn
#         if not grad_fn.name() == "BackwardHookFunctionBackward":
#             raise RuntimeError("Error while setting up backward hooks. Please open "
#                                "an issue with a code sample to reproduce this.")
#
#         fn(grad_fn)
#
#         arg_list = list(args)
#         for idx, val in zip(tensors_idx, new_tensors):
#             arg_list[idx] = val
#
#         return tuple(arg_list), tensors_idx
#
#     def setup_input_hook(self, args):
#         def fn(grad_fn):
#             for hook in self.user_hooks:
#                 self._set_user_hook(grad_fn, hook)
#
#         res, input_idx = self._apply_on_tensors(fn, args)
#         self.n_inputs = len(args)
#         self.input_tensors_index = input_idx
#         return res
#
#     def setup_output_hook(self, args):
#         def fn(grad_fn):
#             def hook(_, grad_output):
#                 self.grad_outputs = self._pack_with_none(self.output_tensors_index,
#                                                          grad_output,
#                                                          self.n_outputs)
#             grad_fn.register_hook(hook)
#
#         is_tuple = True
#         if not isinstance(args, tuple):
#             args = (args,)
#             is_tuple = False
#
#         res, output_idx = self._apply_on_tensors(fn, args)
#         self.n_outputs = len(args)
#         self.output_tensors_index = output_idx
#
#         if not is_tuple:
#             res = res[0]
#         return res




# def _is_not_trainable_variable(obj):
#     return module._is_variable(obj) and not getattr(obj, "trainable", False)


# The internal graph maintained by Keras and used by the symbolic Keras APIs
# while executing eagerly (such as the functional API for model-building).
# This is thread-local to allow building separate models in different threads
# concurrently, but comes at the cost of not being able to build one model
# across threads.




# A global dictionary mapping graph objects to an index of counters used
# for various layer/optimizer names in each graph.
# Allows to give unique autogenerated names to layers, in a graph-specific way.
PER_GRAPH_OBJECT_NAME_UIDS = weakref.WeakKeyDictionary()


def reset_name(module, prefix_dict=None):
    def get_uid(prefix, seq):
        if prefix not in module._uid_prefixs or seq < module._uid_prefixs[prefix]:
            module._uid_prefixs[prefix] = seq
        return module._uid_prefixs[prefix]

    if not hasattr(module, '_uid_prefixs') or prefix_dict is not None:
        module._uid_prefixs = prefix_dict
    if not hasattr(module, 'default_name'):
        module.default_name = camel2snake(module.__class__.__name__) + '_' + str(get_global_uid(camel2snake(module.__class__.__name__)))
    prefix, seq = module.default_name.rsplit('_', 1)  # if '_' in module.default_name else
    seq = int(seq)
    module.default_name = prefix + '_' + str(seq - get_uid(prefix, seq) + 1)
    if module._name is None:
        module._name = module.default_name
    module.__name__ = module._name


class PRNG(object):
    """Just a stateful wrapper for a jax.random.PRNGKey."""
    def __init__(self, key):
        self.key = key
    def split(self):
        (self.key, subkey) = jax.random.split(self.key)
        return subkey

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

_grad_t = Union[Tuple[Tensor, ...], Tensor]
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar('T', bound='Layer')


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
        hook: Callable[['Layer', _grad_t, _grad_t], Union[None, Tensor]]
) -> RemovableHandle:
    r"""Registers a backward hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

        The current implementation will not have the presented behavior
        for complex :class:`Module` that perform many operations.
        In some failure cases, :attr:`grad_input` and :attr:`grad_output` will only
        contain the gradients for a subset of the inputs and outputs.
        For such :class:`Module`, you should use :func:`torch.Tensor.register_hook`
        directly on a specific input or output to get the required gradients.

    The hook will be called every time the gradients with respect to module
    inputs are computed. The hook should have the following signature::

        hook(module, grad_input, grad_output) -> Tensor or None

    The :attr:`grad_input` and :attr:`grad_output` may be tuples if the
    module has multiple inputs or outputs. The hook should not modify its
    arguments, but it can optionally return a new gradient with respect to
    input that will be used in place of :attr:`grad_input` in subsequent
    computations. :attr:`grad_input` will only correspond to the inputs given
    as positional arguments.

    Global hooks are called before hooks registered with `register_backward_hook`

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
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
    raise NotImplementedError

# def Parameter(data, trainable=True, dtype=None, name=None, **kwargs):
#     if dtype is None:
#         dtype = tf.float32
#     return tf.Variable(initial_value=cast(data, dtype), trainable=trainable, name=name)


class Parameter(object):
    """A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~jax.Tensor` subclasses, that have a
    very special property when used with :class:`Layer` s - when they're
    assigned as Layer attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Layer.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor): parameter tensor.
        trainable (bool, optional): whether the parameter is trainable. Default: `True`
        name(string, optional): parameter name

    """
    def __init__(self, data:Tensor, trainable=True, name=None):
        self._data=data
        self._trainable=trainable
        self.name=name

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if self._trainable:
            self._data = value

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(), self._trainable,self.name)
            memo[id(self)] = result
            return result

    def __repr__(self):
        if self.name is not None:
            return f'<Param at {self.name}>'
        else:
            return super().__repr__()





class Layer(object):
    """Trident base layer class, just like nn.Module in pytorch

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
    _parameters: Dict[str, Optional[Parameter]]
    _buffers: Dict[str, Optional[Tensor]]
    _non_persistent_buffers_set: Set[str]
    _backward_hooks: Dict[int, Callable]
    _is_full_backward_hook: Optional[bool]
    _forward_hooks: Dict[int, Callable]
    _forward_pre_hooks: Dict[int, Callable]
    _state_dict_hooks: Dict[int, Callable]
    _load_state_dict_pre_hooks: Dict[int, Callable]
    _load_state_dict_post_hooks: Dict[int, Callable]
    _modules: Dict[str, Optional['Module']]

    def __init__(self,
                 name=None,
                 keep_output=False,
                 device=None,
                 dtype=None ,
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
        super().__setattr__('_backward_hooks', OrderedDict())
        super().__setattr__('_is_full_backward_hook', None)
        super().__setattr__('_forward_hooks', OrderedDict())
        super().__setattr__('_forward_pre_hooks', OrderedDict())
        super().__setattr__('_state_dict_hooks', OrderedDict())
        super().__setattr__('_load_state_dict_pre_hooks', OrderedDict())
        super().__setattr__('_load_state_dict_post_hooks', OrderedDict())
        super().__setattr__('_modules', OrderedDict())
        super().__setattr__('factory_kwargs',{'device': get_device(), 'dtype': Dtype.float32})
        self.to(self.factory_kwargs['device'])



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
        self.in_sequence = kwargs.get('in_sequence', False)
        if self.in_sequence:
            self.filter_index = -1

        self.rank = kwargs.get('rank', None)
        self._nodes = OrderedDict()
        self._input_shape: Optional[None, TensorShape, List[TensorShape]] = None
        self._output_shape: Optional[None, TensorShape, List[TensorShape]] = None


        # self._modules = OrderedDict()
        # self._parameters = OrderedDict()
        # self._buffers = OrderedDict()
        # self._backward_hooks = OrderedDict()
        # self._forward_hooks = OrderedDict()
        # self._forward_pre_hooks = OrderedDict()
        # self._state_dict_hooks = OrderedDict()
        # self._load_state_dict_pre_hooks = OrderedDict()

        # self._non_persistent_buffers_set = set()
        self._input_shape = None
        self._output_shape = None

        self.input_filters = None
        self.input_spec = None

        self.keep_output = keep_output
        self._output_tensor = None
        self._signature = None

        self.dump_patches = True

        self._device = get_device()

    # Trick mypy into not applying contravariance rules to inputs by defining
    # forward as a value, rather than a function.  See also
    # https://github.com/python/mypy/issues/8795
    forward: Callable[..., Any] = _forward_unimplemented


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

    def add_module(self, name, module):
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
                mod._device = self._device
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
            self.add_module(str(len(self._modules)), module)  # self.nodes = nodes  # for mod in self.modules():  #     mod.nodes = nodes

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
        print('Your model will start to rebuild, it will cause lost all existing trainable parameters, will you want to rebuild it?')
        ans = input('(Y/N) << ').lower()
        if ans in ['yes', 'y']:
            for name, module in self.named_modules():
                if module.trainable == True:
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

    def register_forward_pre_hook(self, hook):
        r"""Registers a forward pre-hook on the module.

        The hook will be called every time before :func:`forward` is invoked.
        It should have the following signature::

            hook(module, input) -> None or modified input

        The hook can modify the input. User can either return a tuple or a
        single modified value in the hook. We will wrap the value into a tuple
        if a single value is returned(unless that value is already a tuple).

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(self._forward_pre_hooks)
        self._forward_pre_hooks[handle.id] = hook
        return handle

    def register_forward_hook(self, hook):
        r"""Registers a forward hook on the module.

        The hook will be called every time after :func:`forward` has computed an output.
        It should have the following signature::

            hook(module, input, output) -> None or modified output

        The hook can modify the output. It can modify the input inplace but
        it will not have effect on forward since this is called after
        :func:`forward` is called.

        Returns:
            :class:`tensorflow.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle

    def _get_name(self):
        return self.__class__.__name__

    def register_buffer(self, name, tensor, persistent: bool = True):
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

            >>> self.register_buffer('running_mean', zeros([5]))

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
        elif tensor is not None and not isinstance(tensor, Tensor) and not is_tensor(tensor):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(tensorflow Tensor or None required)".format(type(tensor).__name__, name))
        else:

            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name, param):
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
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(Parameter or None required)".format(type(param).__name__, name))
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
                            module._parameters[name].assign(identity(cast(para.value(), dtype)))
                if module._buffers is not None and len(module._buffers) > 0:
                    for name, buff in module._buffers.items():
                        if buff is None:
                            module._buffers[name] = None
                        else:
                            module._buffers[name] = identity(cast(buff, dtype))
                module._device = self._device
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

        if is_gpu_available:
            if self.get_root()._device != 'cuda:0':
                self.get_root()._device = 'cuda:0'
            return self._apply(lambda t: jax.device_put(t, device=jax.devices("gpu")[0]))

        else:
            sys.stderr.write('GPU is not available in this machone./n')

    def cpu(self: T) -> T:
        r"""Moves all model parameters and buffers to the CPU.

        Returns:
            Module: self
        """
        if self.get_root()._device != 'cpu':
            self.get_root()._device = 'cpu'
        return self._apply(lambda t:jax.device_put(t, device=jax.devices("cpu")[0]))

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

        if is_gpu_available:
            if self.get_root()._device != 'cuda:0':
                self.get_root()._device = 'cuda:0'
            return self._apply(lambda t: jax.device_put(t, device=jax.devices("gpu")[0]))
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
                if isinstance(param, Parameter):
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
        return self._device

    @device.setter
    def device(self, value):
        if isinstance(value, str):
            self._device = value
            self.to(value)

        elif isinstance(value,  jax._src.lib.xla_client.Device):
            self._device = value.__str__()
            self.to(self._device)
        else:
            print(value)

    @property
    def built(self):
        return self._built

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value):

        if is_tensor(value) and value.ndim == 1 and value.dtype == Dtype.int32:
            value = TensorShape([None, ] + to_list(to_numpy(value)))
        elif isinstance(value, (list, tuple)) and len(value) > 0 and all([isinstance(item, numbers.Integral) for item in value]):
            value = TensorShape((None,) + value)
        elif isinstance(value, (list, tuple)) and len(value) > 0 and all([is_tensor(item) and ndim(item) == 1 and item.dtype == Dtype.int32 for item in value]):
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
                        self._signature.inputs['input_{0}'.format(k)] = TensorSpec(shape=self._input_shape[k], name='input_{0}'.format(k))

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
            elif isinstance(value, (list, tuple)) and len(value) > 0 and all([isinstance(item, numbers.Integral) for item in value]):
                value = TensorShape((None,) + value)
            elif isinstance(value, (list, tuple)) and len(value) > 0 and all([is_tensor(item) and ndim(item) == 1 and item.dtype == Dtype.int32 for item in value]):
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
                    self._signature.outputs['output'] = TensorSpec(shape=TensorShape(to_list(to_numpy(self._output_shape))), name='output')
                else:
                    for k in range(len(self._output_shape)):
                        self._signature.outputs['output_{0}'.format(k)] = TensorSpec(shape=self._output_shape[k], name='output_{0}'.format(k))


    @property
    def signature(self):
        """

        Returns:

        """

        arg_spec = get_args_spec(self.forward)
        inspect_args = [arg for arg in list(arg_spec.args) if arg not in ['self', 'kwargs']]
        if isinstance(arg_spec.varargs, str):
            inspect_args.append(arg_spec.varargs)
        inspect_args=unpack_singleton(inspect_args)

        if self._signature is None or len(self._signature) == 0 or len(self._signature.inputs) == 0:
            self._signature = Signature(name=self.name)


            if self._input_shape is not None:
                if isinstance(self._input_shape, TensorShape) and isinstance(inspect_args,str):
                    self._signature.inputs[inspect_args] = TensorSpec(shape=TensorShape(self._input_shape), name=inspect_args)

                elif isinstance(self._input_shape, tuple):
                    for i in range(len(self._input_shape)):
                        self._signature.inputs["input_{0}".format(i)] = TensorSpec(shape=TensorShape(self._input_shape[i]), name="input_{0}".format(i))
            else:
                for arg in inspect_args:
                    self._signature.inputs[arg] = TensorSpec(shape=None)

            if self._output_shape is not None:
                if isinstance(self._output_shape, TensorShape):
                    self._signature.outputs["output"] = TensorSpec(shape=TensorShape(self._output_shape), name="output")
                elif isinstance(self._output_shape, tuple):
                    for i in range(len(self._output_shape)):
                        self._signature.outputs["output_{0}".format(i)] = TensorSpec(shape=to_tensor(self._output_shape[i]), name="output_{0}".format(i))
            else:
                self._signature.outputs["output"] =TensorSpec(shape=None)
        if isinstance(inspect_args,str) and len(self._signature.inputs)==1 and self._signature.inputs.key_list[0] !=inspect_args:
            self._signature.inputs[inspect_args]=self._signature.inputs.value_list[0]
            self._signature.inputs.pop(self._signature.inputs.key_list[0] )
        elif isinstance(inspect_args,list) and len(self._signature.inputs)==len(inspect_args):
            for  k1,k2 in zip(inspect_args,self._signature.inputs.key_list.copy()):
                if k1!=k2:
                    self._signature.inputs[k1]=self._signature.inputs[k2]
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
                    setattr(shadow, k, copy.deepcopy(v))
        shadow.load_state_dict(self.state_dict())
        shadow.to(self.device)
        return shadow

        # The user can pass an optional arbitrary mappable object to `state_dict`, in which case `state_dict` returns
        # back that same object. But if they pass nothing, an `OrederedDict` is created and returned.

    T_destination = TypeVar('T_destination', bound=Mapping[str, Tensor])

    @overload
    def state_dict(self, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination:
        ...

    # TODO: annotate with OrderedDict not Dict, but there is a problem:
    # https://docs.python.org/3/library/typing.html#typing.OrderedDict
    @overload
    def state_dict(self, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Tensor]:
        ...

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

    @overload
    def state_dict(self, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination:
        ...

    # TODO: annotate with OrderedDict not Dict, but there is a problem:
    # https://docs.python.org/3/library/typing.html#typing.OrderedDict
    @overload
    def state_dict(self, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Tensor]:
        ...

    def state_dict(self, destination=None, prefix='', keep_vars=False):
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


    def _call_impl(self, *input, **kwargs):
        forward_call = self.forward

        # Do not call functions when jit is used
        full_backward_hooks, non_full_backward_hooks = [], []
        if self._backward_hooks or _global_backward_hooks:
            full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()

        is_all_numpy = False
        is_built = self._built

        if self.is_root:
            if isinstance(input, (tuple)):
                is_all_numpy = all([isinstance(inp, np.ndarray) for inp in input])
                input = tuple([to_tensor(inp, device=get_device()) for inp in input])
            else:
                if isinstance(input, np.ndarray):
                    is_all_numpy = True
                input = to_tensor(input, device=get_device())
                input = (input,)

        if _global_forward_pre_hooks or self._forward_pre_hooks:
            for hook in (*_global_forward_pre_hooks.values(), *self._forward_pre_hooks.values()):
                result = hook(self, input)
                if result is not None:
                    if not isinstance(result, tuple):
                        result = (result,)
                    input = result




        if not self._built:
            inp = unpack_singleton(input)
            if is_tensor(inp):
                shp = tensor_to_shape(inp, need_exclude_batch_axis=True)
                self.input_filters = shp[self.filter_index]
                self.input_shape = shp
                if self.is_root:
                    if self._signature is None:
                        self._signature = get_signature(self)
                    if self._signature is not None and len(self._signature.inputs) > 0:
                        self._signature.inputs[self._signature.inputs.key_list[0]].shape = tensor_to_shape(inp, need_exclude_batch_axis=True, is_singleton=False)
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

        bw_hook = None
        # if full_backward_hooks:
        #     bw_hook = hooks.BackwardHook(self, full_backward_hooks)
        #     input = bw_hook.setup_input_hook(input)


        # don't use result = self.forward(i*nput, **kwargs) because EagerTensor will splited as a tuple....
        try:

            result = self.forward(*input, **kwargs)
            result = unpack_singleton(result)

            if hasattr(self, 'keep_output') and self.keep_output == True:
                # make a op
                self._output_tensor = result
            if self._output_shape is None or is_built == False:
                output = result
                if is_tensor(output):  # one output
                    self._output_shape = tensor_to_shape(output)
                elif isinstance(output, (list, tuple)):
                    output_shape = tuple(
                        [tensor_to_shape(item) for item in output if not isinstance(item, (list, tuple))])
                    # if not isinstance(item, (list,tuple)) lstm
                    self._output_shape = unpack_singleton(output_shape)

            if _global_forward_hooks or self._forward_hooks:
                for hook in (*_global_forward_hooks.values(), *self._forward_hooks.values()):
                    hook_result = hook(self, input, result)
                    if hook_result is not None:
                        result = hook_result

            if bw_hook:
                result = bw_hook.setup_output_hook(result)

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

    __call__ : Callable[..., Any] = _call_impl

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
        if isinstance(value, Parameter):
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
            if isinstance(value, Layer):
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
                    if value is not None and not isinstance(value, Tensor):
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
    def trainable_weights(self) -> List[Parameter]:
        r"""The list of trainable variables (parameters) of the module.
        Parameters of this module and all its submodules are included.

        .. note::
            The list returned may contain duplicate parameters (e.g. output
            layer shares parameters with embeddings). For most usages, it's not
            necessary to ensure uniqueness.

        """
        return [x for x in self.parameters() if x.trainable]

    @property
    def non_trainable_weights(self) -> List[Parameter]:
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
                need_update = True
        if not need_update:
            print('no parameter trainable state is changed')
        elif value:
            print('{0} parameters have set trainable'.format(n))
        else:
            print('{0} parameters have set untrainable'.format(n))

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should reimplement
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
