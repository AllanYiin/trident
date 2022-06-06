from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import builtins
import functools
import gc
import inspect
import itertools
import logging
import numbers
import operator
import os
import shutil
import sys
import uuid
import warnings
from collections import defaultdict, abc
from types import MethodType
from typing import List, Tuple, Optional, Union, Callable, Any, Iterable,Iterator,Mapping, TypeVar,overload
from functools import partial
import numpy as np

try:
    import _pickle as pickle
except:
    import pickle
from itertools import islice
from distutils.version import LooseVersion
import torch.nn as nn
import torch.onnx
import torch.utils.hooks as hooks
from torch.utils.hooks import RemovableHandle
from torch._jit_internal import _copy_to_script_wrapper
from torch.nn.parameter import Parameter
from trident.backend import common
from trident.backend.common import to_list, addindent, camel2snake, unpack_singleton, enforce_singleton, OrderedDict, get_session, set_session, get_session_value, PrintException, Signature, TensorShape, get_args_spec,is_instance
from trident.backend.tensorspec import *
from trident.backend import iteration_tools
from trident.backend.pytorch_ops import *
from trident.backend import pytorch_ops as tops
from trident.backend import dtype
from trident import context
from trident.context import split_path, make_dir_if_need, sanitize_path


ctx = context._context()
_backend = ctx.get_backend()

__all__ = ['get_device', 'set_device', 'Layer', 'Sequential', 'ModuleList', 'Parameter', 'ModuleDict', 'print_network', 'summary', 'load', 'save', 'Combine',
           'try_map_args_and_call',
           'print_mem_stack',
           'normalize_padding', 'fix_layer', 'DTYPE_MAPPING','fix_pytorch_module']

_FUN_NAMES = [
    ('equal', tops.equal)]
for target_fun_name, source_fun in _FUN_NAMES:
    setattr(Tensor, target_fun_name, source_fun)

version = torch.__version__
sys.stdout.write('Pytorch version:{0}.\n'.format(version))

pt_version = LooseVersion(vstring=version)
base_version = LooseVersion(vstring='1.4.0')
amp_version = LooseVersion(vstring='1.6.0')

if pt_version.version < base_version.version:
    raise ValueError('Not support Pytorch older then version 1.4')
elif pt_version.version >= amp_version.version:
    set_session('amp_available', True if torch.cuda.is_available() and pt_version >= amp_version else False)
    if get_session_value('amp_available'):
        sys.stdout.write('Automatic Mixed Precision Support:{0}.\n'.format(True))
    else:
        sys.stdout.write('Automatic Mixed Precision Support:{0}.\n'.format(False))

DTYPE_MAPPING = {
    torch.bool: dtype.bool,
    torch.int8: dtype.int8,
    torch.int16: dtype.int16,
    torch.int32: dtype.int32,
    torch.int64: dtype.int64,
    torch.uint8: dtype.uint8,
    torch.float16: dtype.float16,
    torch.float32: dtype.float32,
    torch.float64: dtype.float64,
    torch.complex64: dtype.complex64,
    torch.complex128: dtype.complex128,
    torch.cfloat: dtype.cfloat
}


def get_device():
    """get current device

    Returns: device string ('cpu', 'cuda)

    """
    if ctx.device is None:
        set_device("cuda" if torch.cuda.is_available() else 'xpu' if is_tpu_available() else  "cpu")

    return get_session().device


def set_device(device=None):

    if device is None:
        if is_gpu_available():
            device='cuda'
        elif is_tpu_available():
            device='xpu'
        else:
            device='cpu'
    device = device.lower().replace('gpu', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Gpu is not available...')
    if device == 'xpu' and not is_tpu_available():
        raise ValueError('Tpu is not available...')
    try:
        device_=device
        if device=='xpu':
            import torch_xla.core.xla_model as xm
            device_ = xm.xla_device()
        set_session('device', device_)

        gcitems = gc.get_objects()
        for i in range(len(gcitems)):
            obj = gcitems[i]
            try:
                if torch.is_tensor(obj) :
                    obj.to(device_)
                elif isinstance(obj, nn.Module):
                    obj.to(device_)
            except Exception:
                pass
    except Exception as e:
        print(e)


if torch.cuda.is_available() and get_device() == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'



def load(f):
    """

    Args:
        f: a file-like object (has to implement :meth:`read`, :meth`readline`, :meth`tell`, and :meth`seek`),
            or a string or os.PathLike object containing a file name

    Returns:

    """

    item = torch.load(f, map_location=torch.device('cpu'))
    if isinstance(item, nn.Module):
        item.eval()
        item.to(get_device())
    return item


def save(obj, f, is_compressed=False):
    """

    Args:
        obj ():
        f: a file-like object (has to implement write and flush) or a string or
           os.PathLike object containing a file name
        is_compressed ():

    Returns:

    """
    torch.save(obj, f, pickle_module=pickle, _use_new_zipfile_serialization=is_compressed)
    return True


def reset_name(module: nn.Module, prefix_dict=None):
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
    module.__name__ = module._name if hasattr(module, '_name') else module.default_name


_UID_PREFIX = defaultdict(int)


def get_global_uid(prefix=''):
    _UID_PREFIX[prefix] += 1
    return _UID_PREFIX[prefix]


_grad_t = Union[Tuple[Tensor, ...], Tensor]
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar('T', bound='Module')

r"""This tracks hooks common to all modules that are executed before/after
calling forward and backward. This is global state used for debugging/profiling
purposes"""
_global_backward_hooks = OrderedDict()
_global_forward_pre_hooks = OrderedDict()
_global_forward_hooks = OrderedDict()


def register_module_forward_pre_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Registers a forward pre-hook common to all modules.

    warning ::

        This adds global state to the `nn.module` module,
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
    handle = hooks.RemovableHandle(_global_forward_pre_hooks)
    _global_forward_pre_hooks[handle.id] = hook
    return handle


def register_module_forward_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Registers a global forward hook for all the modules

    warning ::

        This adds global state to the `nn.module` module,
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time after :func:`forward` has computed an output.
    It should have the following signature::

        hook(module, input, output) -> None or modified output

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the output. It can modify the input inplace, but
    it will not have effect on forward since this is called after
    :func:`forward` is called.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    This hook will be executed before specific module hooks registered with
    ``register_forward_hook``.
    """
    handle = hooks.RemovableHandle(_global_forward_hooks)
    _global_forward_hooks[handle.id] = hook
    return handle


def register_module_backward_hook(
        hook: Callable[['Module', _grad_t, _grad_t], Optional[Tensor]]
) -> RemovableHandle:
    r"""Registers a backward hook common to all the modules.

    warning ::
        This adds global state to the `nn.module` module,
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
    handle = hooks.RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle


class Parameter(nn.Parameter):
    def __new__(cls, data, trainable=True):
        inst = nn.Parameter.__new__(cls, data, trainable)
        return inst

    @property
    def trainable(self):
        return self.requires_grad

    @trainable.setter
    def trainable(self, value):
        self.requires_grad = value


class Layer(nn.Module):
    """Trident extened pytorch nn.Module as base layer class.

    Your models should also subclass of this class.
    Layer contains :
        modules: another child layer(module) in it.
        parameters: the trainable parameters in the layer.
        buffers: the other non_trainable tensor in the layer.


    Attributes :
        training (bool): If True, means in the training phase, else in the evaluation phase.

        rank (int): The number of the spatial related axes.

        _modules (OrderedDict) : storage of all the submodules.

        _parameters (OrderedDict) : storage of all the tranable weights.

        _buffers (OrderedDict) : storage of all the non-trainable tensor.

        _forward_hooks (OrderedDict) : storage of all the hooks triggered before the forward execution.

        _forward_pre_hooks (OrderedDict) : storage of all the hooks triggered  after the forward execution.

        _state_dict_hooks (OrderedDict) : storage of all the hooks triggered  when state_dict generating  execution.

        _load_state_dict_pre_hooks (OrderedDict) : storage of all the hooks triggered  when loading state_dict   execution.


        input_filters (int): input channels

        signature (int): the function signature of this layer.

        default_name: default_name is the same concept as in keras, it comes from class name with sequence number.

        relative_name:relative_name is the same concept as named_modules in pytorch. But in pytorch, you need to get the name from generator enumeration. In trident,
        you can access the relative name  with this attribute.




    """
    _version: int = 1
    training: bool
    _is_full_backward_hook: Optional[bool]
    def __init__(self, name=None, keep_output=False, **kwargs):
        """

        Args:
            name (str) :name of the layer.
            keep_output (bool) :whether you need to kept output tensor in execution time.


        """
        super(Layer, self).__init__()
        self.batch_index = 0
        self.filter_index = 1
        self.in_sequence = kwargs.get('in_sequence', False)
        if self.in_sequence:
            self.filter_index = -1
        self.training = True
        self._built = False
        self.rank = kwargs.get('rank', None)
        self._non_persistent_buffers_set = set()
        self.uuid = uuid.uuid4().node
        self._nodes = OrderedDict()
        self._uid_prefixs = {}
        self._name = name
        self.is_root = True

        prefix = self.__class__.__name__
        self.default_name = camel2snake(prefix) + '_' + str(get_global_uid(camel2snake(prefix)))
        self.relative_name = ''
        reset_name(self, self._uid_prefixs)
        self._input_shape: Optional[None, TensorShape, List[TensorShape]] = None
        self._output_shape: Optional[None, TensorShape, List[TensorShape]] = None

        self.input_filters = None

        self.keep_output = keep_output
        self.register_buffer('_output_tensor', None, persistent=False)

        self._signature = None
        self._signature=get_signature(self,name=self.name)
        self._device = get_device()


        self.dump_patches = True

    # Trick mypy into not applying contravariance rules to input by defining
    # forward as a value, rather than a function.  See also
    # https://github.com/python/mypy/issues/8795
    def _forward_unimplemented(self, *input: Any, **kwargs) -> None:
        raise NotImplementedError

    r"""Defines the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    forward: Callable[..., Any] = _forward_unimplemented

    def get_root(self):
        if not hasattr(self, '_nodes') or self._nodes is None:
            self.is_root = True
            return self
        elif len(self._nodes) > 0 and list(self._nodes.values())[0].is_root:
            return list(self._nodes.values())[0]
        else:
            for name, node in self._nodes.items():
                if node.is_root:
                    return node
            return self

    # def forward(self, *input,**kwargs):
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

    @property
    def name(self) -> str:
        """If not assign name , it will return the default_name"""
        return self._name if self._name is not None and len(self._name) > 0 else self.relative_name

    @name.setter
    def name(self, value):
        self._name = value
        self.__name__ = value

    @property
    def nodes(self):
        """The whole tree structured OrderedDict { uuid : module } , for module to access any node in these structures, ex. Shortcut"""
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
        if not isinstance(module, (nn.Module, Layer)) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                torch.typename(module)))
        if isinstance(module, (Combine)) and module is not None:
            raise TypeError("{} cannot be added".format(
                torch.typename(module)))
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("module name should be a string. Got {}".format(torch.typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            # name=name.replace('.','_')
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
                mod.to(self._device)
                reset_name(mod, self._uid_prefixs)
                mod.relative_name = name if mod.relative_name == '' else name + '.' + mod.relative_name

    def add(self, module):
        """Simplified add_module

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
            input_shape (TensorShape):  the shape representation exclude the batch axis.

        """
        pass

    def rebuild(self, input_shape):
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
                if len(module._parameters) > 0 or module.trainable:
                    module._input_shape = None
                    module._output_shape = None
                    module._built = False
                    for k in module._parameters.keys():
                        module._parameters[k] = None

                if len(module._forward_pre_hooks) > 0:
                    module._forward_pre_hooks = OrderedDict()

            dummay_input = to_tensor(np.random.standard_normal((2,) + input_shape).astype(np.float32)).to(get_device())
            out = self.forward(dummay_input)

    @property
    def trainable_weights(self) -> List[nn.Parameter]:
        """The list of trainable variables (parameters) of the module.
        Parameters of this module and all its submodules are included.

        Notes:
            The list returned may contain duplicate parameters (e.g. output
            layer shares parameters with embeddings). For most usages, it's not
            necessary to ensure uniqueness.

        """
        paras = []
        for p in self.parameters():
            if p.requires_grad:
                paras.append(p)

        return paras

    @property
    def non_trainable_weights(self) -> List[nn.Parameter]:
        """
        The list of non-trainable variables (parameters) of the module.Parameters of this module and all its submodules are included.

        Notes:
            The list returned may contain duplicate parameters (e.g. output
            layer shares parameters with embeddings). For most usages, it's not
            necessary to ensure uniqueness.

        """
        paras = []
        for p in self.parameters():
            if not p.requires_grad:
                paras.append(p)

        return paras

    @property
    def weights(self):
        """The list of all variables (parameters) of the module.Parameters of this module and all its submodules are included."""
        return list(self.parameters())

    def get_weights(self):
        """The list of all numpy variables ndarray equivelent of the module.Parameters of this module and all its submodules are included."""
        return [p.numpy() for p in list(self.parameters())]

    def set_weights(self, weights):
        params = self.weights
        if not params:
            return
        weight_value_tuples = []
        param_values = [w.data for w in params]
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise ValueError('Layer weight shape ' + str(pv.shape) + ' not compatible with '
                                                                         'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        for p, w in weight_value_tuples:
            p.data = w.data

    @property
    def trainable(self):
        if len(self._parameters) == 0:
            return False

        elif len(self._parameters) > 0:
            for k, v in self._parameters.items():
                if (v is not None and v.requires_grad == False):
                    return False
                elif v is None:
                    pass
                else:
                    pass
            return True

    @trainable.setter
    def trainable(self, value: bool):
        n = 0
        for name, para in self.named_parameters():
            if (para.requires_grad != value):
                para.requires_grad = value
                n += np.prod(list(para.size()))
        if value:
            print('{0} parameters have set trainable'.format(n))
        else:
            print('{0} parameters have set untrainable'.format(n))

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str):
        if isinstance(value, str):
            self._device = value
            self.to(value)
        elif isinstance(value, torch.device):
            self._device = value.type
            self.to(value.type)
        else:
            self._device = value
            self.to(value)

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
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
        return self._apply(lambda t: t.cuda(device))

    def xpu(self, device: Optional[Union[int, torch.device]] = None):
        r"""Moves all model parameters and buffers to the XPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on XPU while being optimized.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        if is_tpu_available() and (device == 'xpu' or device is None):
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        return self._apply(lambda t: t.xpu(device))

    def cpu(self):
        r"""Moves all model parameters and buffers to the CPU.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cpu())

    def gpu(self, device: Optional[Union[int, torch.device]] = None):
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
        return self._apply(lambda t: t.cuda(device))



    # def to(self, *args, **kwargs):
    #     r"""Moves and/or casts the parameters and buffers.
    #
    #     This can be called as
    #
    #     .. function:: to(device=None, dtype=None, non_blocking=False)
    #
    #     .. function:: to(dtype, non_blocking=False)
    #
    #     .. function:: to(tensor, non_blocking=False)
    #
    #     .. function:: to(memory_format=torch.channels_last)
    #
    #     Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
    #     floating point or complex :attr:`dtype`s. In addition, this method will
    #     only cast the floating point or complex parameters and buffers to :attr:`dtype`
    #     (if given). The integral parameters and buffers will be moved
    #     :attr:`device`, if that is given, but with dtypes unchanged. When
    #     :attr:`non_blocking` is set, it tries to convert/move asynchronously
    #     with respect to the host if possible, e.g., moving CPU Tensors with
    #     pinned memory to CUDA devices.
    #
    #     See below for examples.
    #
    #     .. note::
    #         This method modifies the module in-place.
    #
    #     Args:
    #         device (:class:`torch.device`): the desired device of the parameters
    #             and buffers in this module
    #         dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
    #             the parameters and buffers in this module
    #         tensor (torch.Tensor): Tensor whose dtype and device are the desired
    #             dtype and device for all parameters and buffers in this module
    #         memory_format (:class:`torch.memory_format`): the desired memory
    #             format for 4D parameters and buffers in this module (keyword
    #             only argument)
    #
    #     Returns:
    #         Module: self
    #
    #     Examples::
    #
    #         >>> linear = nn.Linear(2, 2)
    #         >>> linear.weight
    #         Parameter containing:
    #         tensor([[ 0.1913, -0.3420],
    #                 [-0.5113, -0.2325]])
    #         >>> linear.to(torch.double)
    #         Linear(in_features=2, out_features=2, bias=True)
    #         >>> linear.weight
    #         Parameter containing:
    #         tensor([[ 0.1913, -0.3420],
    #                 [-0.5113, -0.2325]], dtype=torch.float64)
    #         >>> gpu1 = torch.device("cuda:1")
    #         >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
    #         Linear(in_features=2, out_features=2, bias=True)
    #         >>> linear.weight
    #         Parameter containing:
    #         tensor([[ 0.1914, -0.3420],
    #                 [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
    #         >>> cpu = torch.device("cpu")
    #         >>> linear.to(cpu)
    #         Linear(in_features=2, out_features=2, bias=True)
    #         >>> linear.weight
    #         Parameter containing:
    #         tensor([[ 0.1914, -0.3420],
    #                 [-0.5112, -0.2324]], dtype=torch.float16)
    #
    #         >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
    #         >>> linear.weight
    #         Parameter containing:
    #         tensor([[ 0.3741+0.j,  0.2382+0.j],
    #                 [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
    #         >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
    #         tensor([[0.6122+0.j, 0.1150+0.j],
    #                 [0.6122+0.j, 0.1150+0.j],
    #                 [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
    #
    #     """
    #
    #     device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
    #
    #     if dtype is not None:
    #         if not (dtype.is_floating_point or dtype.is_complex):
    #             raise TypeError('nn.Module.to only accepts floating point or complex '
    #                             'dtypes, but got desired dtype={}'.format(dtype))
    #         if dtype.is_complex:
    #             warnings.warn(
    #                 "Complex modules are a new feature under active development whose design may change, "
    #                 "and some modules might not work as expected when using complex tensors as parameters or buffers. "
    #                 "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.md "
    #                 "if a complex module does not work as expected.")
    #
    #     def convert(t):
    #         if convert_to_format is not None and t.dim() in (4, 5):
    #             return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
    #                         non_blocking, memory_format=convert_to_format)
    #         return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
    #
    #     return self._apply(convert)

    @property
    def built(self):
        return self._built

    @property
    def input_shape(self):
        """Shape of input tensor,not including the batch axis."""
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value) -> TensorShape:
        """ Setting the input_shape, means the layer get shape information and start to do the shape inferrence """

        if is_tensor(value) and value.ndim == 1 and value.dtype == torch.int32:
            value = TensorShape([None, ] + to_list(to_numpy(value)))
        elif isinstance(value, torch.Size):
            dims = [None, ] + [d for d in value]
            value = TensorShape(dims)
        elif isinstance(value, (list, tuple)) and len(value) > 0 and all([isinstance(item, numbers.Integral) for item in value]):
            value = TensorShape((None,) + value)
        elif isinstance(value, (list, tuple)) and len(value) > 0 and all([is_tensor(item) and ndim(item) == 1 and item.dtype == torch.int32 for item in value]):
            value = [TensorShape(sh) for sh in value]
        elif isinstance(value, TensorShape):
            pass
        else:
            value = TensorShape(value)

        if self._built == False or self._input_shape is None or self.input_filters is None:
            self._input_shape = value
            self.input_filters = self._input_shape[self.filter_index]
            self.build(value)
            self._built = True

    @property
    def input_spec(self):
        if self.is_root and self._signature is not None:
            return unpack_singleton(self._signature.inputs.value_list)

    @property
    def output_shape(self):
        return self._output_shape

    @output_shape.setter
    def output_shape(self, value):
        if is_tensor(value) and value.ndim == 1 and value.dtype == torch.int32:
            value = TensorShape([None, ] + to_list(to_numpy(value)))
        elif isinstance(value, torch.Size):
            dims = [None, ] + [d for d in value]
            value = TensorShape(dims)
        elif isinstance(value, (list, tuple)) and len(value) > 0 and all([isinstance(item, numbers.Integral) for item in value]):
            value = TensorShape((None,) + value)
        elif isinstance(value, (list, tuple)) and len(value) > 0 and all([is_tensor(item) and ndim(item) == 1 and item.dtype == torch.int32 for item in value]):
            value = [TensorShape(sh) for sh in value]
        elif isinstance(value, TensorShape):
            pass
        else:
            value = TensorShape(value)

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
        inspect_args = unpack_singleton(inspect_args)

        if self._signature is None or len(self._signature) == 0 or len(self._signature.inputs) == 0:
            self._signature = Signature(name=self.name)

            if self._input_shape is not None:
                if isinstance(self._input_shape, TensorShape) and isinstance(inspect_args, str):
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

    # @property
    # def input(self):
    #     return NotImplemented

    @property
    def output(self):
        """
            Retrieves the output tensor(s) of a layer.
            for memory saving issue, we don't prefer to keep every input/output
            tensor in every layer.You should set self.keep_output flag to True, and then
            retrive the output tensor when the calll() is executing.
        Returns
            Output tensor or list of output tensors.

        """
        if not self.keep_output:
            return None
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

    def save_onnx(self, file_path=''):
        input_shape = self._signature.inputs.value_list[0].shape.dims
        input_shape[0] = 1
        self.eval()
        x = cast(torch.randn(*input_shape, requires_grad=False), self.input_spec.dtype)
        _out = self(x)
        folder, filename, ext = split_path(file_path)
        if filename == '':
            filename= self.name
        ext = '.onnx_'
        save_path = sanitize_path(os.path.join(folder, filename + ext))
        make_dir_if_need(save_path)

        # Export the model
        torch.onnx.export(self,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          save_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=11,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                        'output': {0: 'batch_size'}})
        self.train()
        shutil.copy(save_path, save_path.replace('.onnx_', '.onnx'))
        os.remove(save_path)

    def _call_impl(self, *input, **kwargs):
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)

        # Do not call functions when jit is used
        full_backward_hooks, non_full_backward_hooks = [], []
        if self._backward_hooks or _global_backward_hooks:
            full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()

        is_all_numpy = False
        is_built = self._built

        # only do in the root
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
        if full_backward_hooks:
            bw_hook = hooks.BackwardHook(self, full_backward_hooks)
            input = bw_hook.setup_input_hook(input)

        # orig_input=[inp.copy().detach() for inp in input]
        # is_abnormal=any([any_abnormal_number(inp)for inp in orig_input])
        # if len(self.weights)>0 and self.weights[0] is not None and not self.weights[0].requires_grad:
        #     print(self.relative_name,'weights requires_grad',self.weights[0].requires_grad)

        result = forward_call(*input, **kwargs)
        # if not is_abnormal and any([any_abnormal_number(inp)for inp in result]):
        #     print('abnormal_number',self.relative_name,'before:',is_abnormal)
        #     print('orig_input',[inp[0,:] for inp in orig_input])
        #     print('result', [inp[0,:] for inp in result])
        #     print('weight',list(self.named_parameters()))
        #     print('activation',self.activation)
        #     out=forward_call(*orig_input, **kwargs)

        result = unpack_singleton(result)


        if hasattr(self, 'keep_output') and self.keep_output == True:
            # make a op
            self._output_tensor = result
        if self._output_shape is None or is_built == False:
            output = result
            if is_tensor(output):  # one output
                self._output_shape = tensor_to_shape(output)
            elif isinstance(output, (list, tuple)):
                output_shape = tuple([tensor_to_shape(item) for item in output if item is not None and not isinstance(item, (list, tuple))])
                # if not isinstance(item, (list,tuple)) lstm
                self._output_shape = unpack_singleton(output_shape)

        if _global_forward_hooks or self._forward_hooks:
            for hook in (*_global_forward_hooks.values(), *self._forward_hooks.values()):
                hook_result = hook(self, input, result)
                if hook_result is not None:
                    result = hook_result

        if bw_hook:
            result = bw_hook.setup_output_hook(result)

        # Handle the non-full backward hooks
        if non_full_backward_hooks:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in non_full_backward_hooks:
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
                self._maybe_warn_non_full_backward_hook(input, result, grad_fn)


        if is_all_numpy == True and self.training == False and self.is_root == True:
            if is_tensor(result):
                return to_numpy(result)
            elif isinstance(result, (list, tuple)):
                result = list(result)
                return tuple([to_numpy(res) if is_tensor(res) else res for res in result])
        return result

    __call__: Callable[..., Any] = _call_impl

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

    def __setattr__(self, name: str, value) -> None:
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
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Layer):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
                value.is_root = False
                for mod in value.modules():
                    if isinstance(mod, Layer) and mod.uuid != value.uuid:
                        mod.is_root = False
                reset_name(value, self._uid_prefixs)
                value.relative_name = name if not hasattr(value, 'relative_name') or value.relative_name == '' else name + '.' + value.relative_name
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch.typename(value), name))
                modules[name] = value
                value.is_root = False
                for mod in value.modules():
                    if isinstance(mod, Layer) and mod.uuid != value.uuid:
                        mod.is_root = False
                reset_name(value, self._uid_prefixs)
                value.relative_name = name if not hasattr(value, 'relative_name') or value.relative_name == '' else name + '.' + value.relative_name
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(torch.typename(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __getstate__(self):
        # Override to support `copy.deepcopy` and pickling.
        # Thread-local objects cannot be copied in Python 3, so pop these.
        # so shouldn't be copied.
        state = self.__dict__.copy()
        # state.pop('_thread_local', None)
        # state.pop('_metrics_lock', None)
        return state

    def __setstate__(self, state):
        # state['_thread_local'] = threading.local()
        # state['_metrics_lock'] = threading.Lock()
        # Bypass Trackable logic as `__dict__` already contains this info.
        object.__setattr__(self, '__dict__', state)

    def __repr__(self):
        # We treat the extra repr like the submodule, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = addindent(mod_str, 2)
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

    def __init__(self, *args, name=None):
        super(Sequential, self).__init__(name=name)
        self._name = name
        self._built = False
        self.uuid = uuid.uuid4().node
        args = unpack_singleton(args)
        if isinstance(args, (dict, OrderedDict, ModuleDict, nn.ModuleDict)):
            for key, module in args.items():
                module.name = key
                self.add_module(key, module)
        elif isinstance(args, (list, tuple, nn.ModuleList, ModuleList)):
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        else:
            for idx, module in enumerate(args):
                if module._name is not None and len(module._name) > 0:
                    self.add_module(module._name, module)
                else:
                    self.add_module(str(idx), module)
        self.to(self.device)

    def build(self, input_shape: TensorShape):
        """

        Args:
            input_shape (torch.Size, tensor, list(int), tuple(int)): The input_shape information, not including batch axis.

        Returns:

        """
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

                self._output_shape = tuple([tensor_to_shape(o, need_exclude_batch_axis=True, is_singleton=False) for o in out.value_list])
                self.get_root().signature.outputs = OrderedDict()
                for k, v in out.item_list:
                    self.get_root().signature.outputs[k] = tensor_to_shape(v, need_exclude_batch_axis=True, is_singleton=False)
            else:
                out = enforce_singleton(out)
                self._output_shape = tensor_to_shape(out, need_exclude_batch_axis=True, is_singleton=False)
                self._signature.outputs[self._signature.outputs.key_list[0]].shape=self._output_shape
                # if len(self.get_root().signature.outputs) > 0:
                #     self.get_root().signature=get_signature(self)
                # else:
                #     self.get_root().signature.outputs['output'] = self._output_shape.copy()

        else:
            if not hasattr(module,'_signature') or module._signature is None:
                module._signature=get_signature(module)
            sig = copy.deepcopy(module._signature)
            super(Sequential, self).add_module(name, module)
            if len(self) == 1 or self._signature is None:
                self._signature = sig
            elif len(self) > 1:
                self._signature.outputs = copy.deepcopy(sig.outputs)


    def remove_at(self, idx):
        self.__delitem__(idx)

        if len(self._modules) > 0:
            self._output_shape = self[-1]._output_shape
            if isinstance(self._signature, Signature):
                self._signature.outputs[self._signature.outputs.key_list[0]].shape=self[-1]._output_shape

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = idx.__index__()
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            returnDict = OrderedDict()
            for k, v in list(self._modules.items())[idx]:
                returnDict[k] = v
            return tuple(returnDict.value_list)
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

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._modules.values())

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
            #class_name=module.__class__.__name__.lower()
            # if 'lstm' in class_name or 'gru' in class_name:
            #     if isinstance(x,tuple):
            #         x,hx=x
            #         kwargs['hx']=hx

        return x


class ModuleList(Layer):
    r"""Holds submodules in a list.

    :class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    Args:
        modules (iterable, optional): an iterable of modules to add

    Examples:

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x, **kwargs):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules=None, name=None, keep_output=False):
        super(ModuleList, self).__init__(name=name, keep_output=keep_output)
        self.uuid = uuid.uuid4().node
        self._name = name
        if isinstance(modules, dict):
            for key, value in modules.items():
                self.add_module(key, value)
        elif isinstance(modules, (list, tuple)):
            for i, value in enumerate(modules):
                self.add_module(str(i), value)

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    @_copy_to_script_wrapper
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            layer = self._modules[self._get_abs_string_index(idx)]
            layer._nodes = OrderedDict()
            return layer

    def __setitem__(self, idx, module):
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index, module):
        r"""Insert a given module before a given index in the list.

        Args:
            index (int): index to insert.
            module (nn.Module): module to insert
        """

        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]

        self._modules[str(index)] = module

    def append(self, module):
        r"""Appends a given module to the end of the list.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        r"""Appends modules from a Python iterable to the end of the list.

        Args:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, abc.Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self




class ModuleDict(Layer):
    r"""Holds submodules in a dictionary.

    :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~torch.nn.ModuleDict` (the argument to :meth:`~torch.nn.ModuleDict.update`).

    Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    def __init__(self, modules: Optional[Mapping[str, Layer]] = None, name=None, keep_output=False, is_multicasting=False, **kwargs) -> None:
        super(ModuleDict, self).__init__(name=name, keep_output=keep_output, **kwargs)
        self.uuid = uuid.uuid4().node
        self.is_multicasting = is_multicasting
        if modules is not None:
            if len(modules) > 0:
                self.update(modules)

    # @_copy_to_script_wrapper
    def __getitem__(self, key: str) -> Layer:
        return self._modules[key]

    def __setitem__(self, key: str, module: Layer) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules)

    @_copy_to_script_wrapper
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

    # @_copy_to_script_wrapper
    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys.
        """
        return self._modules.keys()

    # @_copy_to_script_wrapper
    def items(self) -> Iterable[Tuple[str, Layer]]:
        r"""Return an iterable of the ModuleDict key/value pairs.
        """
        return self._modules.items()

    # @_copy_to_script_wrapper
    def values(self) -> Iterable[Layer]:
        r"""Return an iterable of the ModuleDict values.
        """
        return self._modules.values()

    def update(self, modules: Mapping[str, Layer]) -> None:
        r"""Update the :class:`~torch.nn.ModuleDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Arguments:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
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
            if self.is_root:
                self._input_shape = input_shape

                dummay_input = to_tensor(self.input_shape.get_dummy_tensor()).to(self.device)

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
        self.uuid = uuid.uuid4().node
        self._name = name
        self._built = False

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        elif len(args) == 1 and isinstance(args[0], (list, nn.ModuleList)):
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
        if not isinstance(module, (nn.Module, Layer)) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                torch.typename(module)))
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("module name should be a string. Got {}".format(torch.typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            # name=name.replace('.','_')
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    def build(self, *input_shape: TensorShape):
        if not self._built:
            # Signature(name=self.name)
            for shp, module in zip(input_shape, self._modules.values()):
                # shp = tensor_to_shape(inp, need_exclude_batch_axis=True, is_singleton=False)
                module.build(shp)
            self._built = True

    @property
    def signature(self):
        """

        Returns:

        """
        if self._signature is None:
            self._signature = Signature(name=self.name)
        for k, v in self._modules.items():
            for inp_k, inp_v in v.signature.inputs.items():
                if '{0}_{1}'.format(inp_k, k) not in self._signature.inputs:
                    self._signature.inputs['{0}_{1}'.format(inp_k, k)] = inp_v
            for out_k, out_v in v.signature.outputs.items():
                if '{0}_{1}'.format(out_k, k) not in self._signature.outputs:
                    self._signature.outputs['{0}_{1}'.format(out_k, k)] = out_v
        return self._signature

    @signature.setter
    def signature(self, value):
        self._signature = value

    def combine_forward(self, x):
        outputs = []
        for inp, module in zip(x, list(self._modules.values())):
            outputs.append(module(inp))
        return tuple(outputs)

    def forward(self, *x):
        if len(x) == len(self._modules.values()):
            outputs = []
            for inp, module in zip(x, list(self._modules.values())):
                outputs.append(module(inp))
            return tuple(outputs)
        elif len(x) == 1:
            outputs = []
            for module in self._modules.values():
                outputs.append(module(x))
            return tuple(outputs)
        return None




def print_network(net, verbose=False):
    num_params = 0
    for i, param in enumerate(net.parameters()):
        num_params += param.numel()
    if verbose:
        logging.info(net)
    logging.info('Total number of parameters: %d\n' % num_params)


def summary(model, input_specs, batch_size=1, inputs=None, device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            # class_name =module.re    module.name   # str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = module.relative_name if hasattr(module, 'relative_name') else module.name
            if m_key in summary:
                summary[m_key]["visits"]+=1
                visit=summary[m_key]["visits"]
                m_key=m_key+'_{0}'.format(visit)

            summary[m_key] = OrderedDict()
            summary[m_key]["visits"]=0
            summary[m_key]["class_name"] = module.__class__.__name__
            if hasattr(module, 'keep_output'):
                summary[m_key]["keep_output"] = module.keep_output
            else:
                summary[m_key]["keep_output"] = False
            input = iteration_tools.flatten([input], iterable_types=(list, tuple))
            input = unpack_singleton([item for item in input if item is not None])

            if isinstance(input, (list, tuple)) and len(input)>0:
                summary[m_key]["input_shape"] = list(int_shape(input[0]))
            elif isinstance(input, dict) and len(input)>0:
                summary[m_key]["input_shape"] = list(int_shape(list(input.values())[0]))
            elif is_tensor(input):
                summary[m_key]["input_shape"] = list(int_shape(input))
            if "input_shape" in summary[m_key]  and  len(summary[m_key]["input_shape"])>0:
                summary[m_key]["input_shape"][0] = batch_size

            output = iteration_tools.flatten([output], iterable_types=(list, tuple))
            output = unpack_singleton([item for item in output if item is not None])
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = list(int_shape(output[0]))
            elif is_instance(output,'OrderedDict'):
                summary[m_key]["output_shape"] = list(int_shape(output.__dict__[list(output.__dict__.keys())[0]]))
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
                    if para.requires_grad:
                        summary[m_key]["trainable"] += num_params

                    summary[m_key]["flops"] += (2 * num_params - 1) * spatial_dims
                    summary[m_key]["macc"] += num_params * spatial_dims

            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, (nn.Sequential, Sequential, nn.ModuleList, ModuleList, nn.ModuleDict, ModuleDict))
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower() if isinstance(device,str) else device.__name__
    assert device in [
        "cuda",
        "cpu",
        "tpu",
        "xla",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    # if device == "cuda" and torch.cuda.is_available():
    #     dtype = torch.cuda.FloatTensor
    # else:
    #     dtype = torch.FloatTensor

    for name, module in model.named_modules():
        module.relative_name = name
    # multiple inputs to the network

    # prevent pytorch 'ValueError: Expected more than 1 value per channel when training, got input size ....
    model.to(get_device())
    model.eval()

    # batch_size of 2 for batchnorm
    # if module.signature is not None:
    #     input_tensor = [to_tensor(module.signature.inputs.value_list[k].get_dummy_tensor()) for k in  range(module.signature.inputs.value_list) if not module.signature.inputs.value_list[k].optional or k==0]
    #
    inps=OrderedDict()
    for v in input_specs:
        k=v.name
        if v.shape is not None and v.shape._dims!=[None]:
            inps[k] =to_tensor(v.get_dummy_tensor(),device=get_device())
        elif v.optional:
            inps[k]=v.default
        elif v.shape is None:
            inps[k]=None
        else:
            inps[k] = None


    #x = [to_tensor(inps[n].get_dummy_tensor(),device=get_device())  for n in range(len(inps)) if inps[n].optional == False or  inps[n].shape._dims!=[None]]
    # p    rint(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    if inputs is not None:
        if isinstance(inputs, OrderedDict):
            model(*list(inps.values()))
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
        max_weight_len = builtins.max(max_weight_len, builtins.max([len(str(item).replace('(', '').replace(')', '')) for item in summary[layer]["weight"].items()]) + 5 if len(
            summary[layer]["weight"]) > 0 else 5)

    print("--------------------------------------------------------------------------------------------------------------------------------")
    line_new = "{0:^50s} {1:<25s}  {2:<35s} {3:<8s}  {4:<8s}  {5:<25s}".replace('50s', str(max_name_len) + 's').replace('35s', str(max_weight_len) + 's').format("Layer (type)",
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

        line_new = "{0:<50s} {1:<25s}  {2:<35s} {3:<8s}  {4:,.0f}  {5:,.0f}  ".replace('50s', str(max_name_len) + 's').replace('35s', str(max_weight_len) + 's').format(
            (layer + "  [" + class_name + "]").ljust(max_name_len, ' '),
            (is_keep + str([None] + summary[layer]["output_shape"][1:])).ljust(25, ' '),
            str(summary[layer]["weight"].item_list[0] if 'weight' in summary[layer] and len(summary[layer]["weight"]) > 0 else ' ').replace('(', '').replace(')', '').ljust(
                max_weight_len, ' '),
            str(summary[layer]["bias"].item_list[0] if 'bias' in summary[layer] and len(summary[layer]["bias"]) > 0 else ' ').replace('(', '').replace(')', '').ljust(8, ' '),
            summary[layer]["nb_params"],
            summary[layer]["flops"].sum()
        )
        if len(summary[layer]["weight"]) > 1:
            for n in range(1, len(summary[layer]["weight"])):
                line_new_add = "{0:<50s} {1:<25s}  {2:<35s} {3:<8s}  {4}  {5}  ".replace('50s', str(max_name_len) + 's').replace('35s', str(max_weight_len) + 's').format(
                    " ".ljust(max_name_len + len(layer + "  [" + class_name + "]") // 2, " "),
                    " ".ljust(25 + len(is_keep + str([None] + summary[layer]["output_shape"][1:])) // 2, " "),
                    str(summary[layer]["weight"].item_list[n] if n < len(summary[layer]["weight"]) else ' ').replace('(', '').replace(')', '').ljust(max_weight_len, " "),
                    str(summary[layer]["bias"].item_list[n] if n < len(summary[layer]["bias"]) else ' ').replace('(', '').replace(')', '').ljust(8, " "), " ", " ")
                line_new = line_new + '\n' + line_new_add

        total_params += summary[layer]["nb_params"]
        flops += float(summary[layer]["flops"])
        macc += float(summary[layer]["macc"].sum())
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            trainable_params += summary[layer]["trainable"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = np.asarray([np.abs(np.prod(to_numpy(spec.shape.dims[1:])) * batch_size * 4. / (1024 ** 2.)) for spec in input_specs if spec.optional == False and spec.shape is not None]).sum()
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


def normalize_padding(padding, rank):
    """
    normalized format of padding should have length equal to rank+2
    And the order should follow the order of dimension
    ex. Conv2d (rank=2) it's normalized format length:2+2  ==>(left, right,top bottom)

    Args:
        padding (None, int, tuple):
        rank (int):

    Returns:
        the normalized format of padding

    Examples:
        >>> normalize_padding(((1,0),(1,0)),2)
        (1, 0, 1, 0)
        >>> normalize_padding((1,0),2)
        (0, 0, 1, 1)
    """
    if padding is None:
        padding = (0,) * (2 * rank)
    elif isinstance(padding, int):
        padding = (padding,) * (2 * rank)
    elif isinstance(padding, (list, tuple)) and len(padding) == 1:
        padding = padding * (2 * rank)
    elif isinstance(padding, (list, tuple)) and len(padding) == rank and isinstance(padding[0], int):
        # rank=2 (1,1)=>(1,1,1,1)   (1,0)=>(0,0,1,1)
        reversed_padding = list(padding)
        reversed_padding.reverse()
        return_padding = []
        for i in range(rank):
            return_padding.append(reversed_padding[i])
            return_padding.append(reversed_padding[i])

        padding = tuple(return_padding)
    elif isinstance(padding, (list, tuple)) and len(padding) == rank and isinstance(padding[0], (list, tuple)):
        # rank=2  ((1,0),(1,0)=>(1,0,1,0)
        padding = tuple(list(itertools.chain(*list(padding))))
    elif isinstance(padding, (list, tuple)) and len(padding) == 2 * rank and isinstance(padding[0], int):
        padding = padding
    return padding


def print_mem_stack():  # pragma: no cover
    """

    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except Exception:
            pass


def count_mem_items():  # pragma: no cover
    nb_params = 0
    nb_tensors = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                obj_type = str(type(obj))
                if 'parameter' in obj_type:
                    nb_params += 1
                else:
                    nb_tensors += 1
        except Exception:
            pass

    return nb_params, nb_tensors


def get_human_readable_count(number):
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    Examples:
        123     -> 123
        1234    -> 1 K       (one thousand)
        2e6     -> 2 M       (two million)
        3e9     -> 3 B       (three billion)
        4e12    -> 4 T       (four trillion)
        5e15    -> 5,000 T
    :param number: a positive integer number
    :returns a string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = [' ', 'K', 'M', 'B', 'T']
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10 ** shift)
    index = num_groups - 1
    return '{0:,d} {1}'.format(int(number),labels[index])


def try_map_args_and_call(fn, data: OrderedDict, data_feed=None, is_autocast_enabled=False):
    """This function is the core function for mapping callable and argments

    Args:

        fn (callable): the callable, maybe functions or layers
        data (OrderedDict): The key-value pair for available data.
        data_feed (OrderedDict): The relation between callable argments (key) and data (value)
        is_autocast_enabled (bool):

    Returns:
        The result of the callable base on data_feed

    """
    if isinstance(fn, torch.Tensor):
        return fn
    else:
        try:
            arg_map = OrderedDict()
            out = None
            if isinstance(fn, Layer):
                _device = fn.get_root().device
                _signature = fn._signature
                if None in fn._signature.inputs.value_list:
                    _signature = get_signature(fn)
                for arg in _signature.inputs.key_list:
                    is_optional = fn.signature.inputs[arg].optional if isinstance(fn.signature.inputs[arg], TensorSpec) else False
                    default = fn.signature.inputs[arg].default if isinstance(fn.signature.inputs[arg], TensorSpec) else None
                    is_input = arg.lower() in ['x', 'input']
                    is_output = arg.lower() in ['y', 'output', 'y_pred']
                    if arg in data_feed and data_feed[arg] in data:
                        arg_map[arg] = to_tensor(data[data_feed[arg]], device=_device)
                    elif arg in data:
                        arg_map[arg] = to_tensor(data[arg], device=_device)
                    elif is_input and 'input' in data_feed and data_feed['input'] in data:
                        arg_map[arg] = to_tensor(data[data_feed['input']], device=_device)
                    elif is_output and 'output' in data_feed and data_feed['output'] in data:
                        arg_map[arg] = to_tensor(data[data_feed['output']], device=_device)
                    elif is_optional:
                        arg_map[arg] = default
                    else:
                        raise ValueError('arg :{0} cannot mapping correctly!'.format(arg))
                # print('arg_map',arg_map.key_list)
                if ctx.amp_available == True and is_autocast_enabled == True and get_device() == 'cuda':
                    with torch.cuda.amp.autocast():
                        out = fn(*arg_map.value_list)
                else:
                    out = fn(*arg_map.value_list)
                    # for item in data.value_list:
                    #     if hasattr(item, 'cpu'):
                    #         item.cpu()
                return out
            elif (hasattr(fn, 'signature') or hasattr(fn, '_signature')) and callable(fn):
                sig = fn._signature if hasattr(fn, '_signature') else fn.signature
                for arg in fn.signature.inputs.key_list:
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
                    elif isinstance(fn, partial) and arg in fn.keywords:
                        arg_map[arg] = fn.keywords[arg]
                    elif is_optional:
                        arg_map[arg] = default
                    else:
                        raise ValueError('arg :{0} cannot mapping correctly!'.format(arg))

                # print('arg_map', arg_map.key_list)
                if ctx.amp_available == True and is_autocast_enabled == True and get_device() == 'cuda':
                    with torch.cuda.amp.autocast():
                        if isinstance(fn, partial):
                            out = fn.func(*arg_map.value_list)
                        else:
                            out = fn(*arg_map.value_list)
                else:
                    if isinstance(fn, partial):
                        out = fn.func(*arg_map.value_list)
                    else:
                        out = fn(*arg_map.value_list)
                    # for item in data.value_list:
                    #     if hasattr(item, 'cpu'):
                    #         item.cpu()

                return out
            elif callable(fn):
                fn.signature = get_signature(fn)
                args = fn.signature.inputs.key_list
                for arg in args:
                    is_optional = fn.signature.inputs[arg].optional if isinstance(fn.signature.inputs[arg], TensorSpec) else False
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
                # print('arg_map', arg_map.key_list)
                if ctx.amp_available == True and is_autocast_enabled == True and get_device() == 'cuda':
                    with torch.cuda.amp.autocast():
                        out = fn(*arg_map.value_list)
                else:
                    out = fn(*arg_map.value_list)
                    # for item in data.value_list:
                    #     if is_tensor(item):
                    #         item.cpu()
                return out
            else:
                print('uncomplete arg_map', arg_map.key_list)
        except Exception as e:
            print(e)
            PrintException()


def force_deterministic(seed):
    """ Force most of the computation nodes to run deterministically.

    Args:
        seed (int): set the random seed for all random ops in the graph and readers.

    """

    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def fix_layer(layer: Layer):
    """fix existing out-of-date model compatibility

    Args:
        layer (trident Layer):

    Returns: fixed layer

    """
    if not hasattr(layer, 'is_root'):
        layer.is_root = True
    if not hasattr(layer, '_device'):
        layer._device = get_device()

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

    if layer._input_shape is not None and isinstance(layer._input_shape, tuple) and all([isinstance(d, numbers.Integral) for d in layer._input_shape]):
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
    elif layer._input_shape is not None and isinstance(layer._input_shape, TensorShape) and is_tensor(layer._input_shape.dims[0]):
        layer._input_shape = TensorShape([d.item() for d in layer._input_shape.dims])

    if layer._output_shape is not None and isinstance(layer._output_shape, tuple) and all([isinstance(d, numbers.Integral) for d in layer._output_shape]):
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
    elif layer._output_shape is not None and isinstance(layer._output_shape, TensorShape) and isinstance(layer._output_shape.dims[0], torch.Size) and len(
            layer._output_shape.dims[0]) == 1:
        layer._output_shape = TensorShape([None if d is None else d[0] for d in layer._output_shape.dims])
    elif layer._output_shape is not None and isinstance(layer._output_shape, TensorShape) and isinstance(layer._output_shape.dims[0], torch.Size):
        layer._output_shape = tuple([TensorShape(to_numpy(d)) for d in layer._output_shape.dims])

    if not hasattr(layer, 'get_root'):
        setattr(layer, 'get_root', MethodType(get_root, layer))

    for module in layer.modules():
        module.dump_patches = True
        if not hasattr(module, '_signature'):
            module._signature = inspect.signature(module.forward)
        class_name = module.__class__.__name__
        if not hasattr(layer, 'get_root'):
            setattr(layer, 'get_root', MethodType(get_root, layer))
        if not hasattr(module, 'uuid'):
            module.uuid = uuid.uuid4().node
        # check for root
        # if module.uuid == layer.uuid:
        #     module.is_root = True
        # else:
        #     module.is_root = False
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

    if layer.is_root == True and (not hasattr(layer, '_signature') or layer._signature is None or len(layer._signature.inputs) == 0):
        layer._signature = None
    return layer


def fix_pytorch_module(module: nn.Module, input_tensor: Tensor = None, input_shape: (tuple, TensorShape) = None):
    module.is_root = True
    module.name=module.__class__.__name__
    module._nodes = OrderedDict()
    module._uid_prefixs = defaultdict(int)
    module._signature = get_signature(module)
    module.signature = get_signature(module)

    def getweights(module):
        return list(module.parameters())

    def get_uid(prefix=''):
        module._uid_prefixs[prefix] += 1
        return module._uid_prefixs[prefix]

    def get_root(module):
        if not hasattr(module, '_nodes') or module._nodes is None or len(module._nodes) < 2:
            return module
        if hasattr(list(module._nodes.values())[0], 'is_root') and list(module._nodes.values())[0].is_root == True:
            return list(module._nodes.values())[0]
        else:
            for name, node in module._nodes.items():
                if hasattr(node, 'default_name') and node.default_name == "sequential_1":
                    return node
            return module


    for name, mod in module.named_modules():
        if mod != module:
            module.is_root = False
        mod._built = True
        if not hasattr(mod,'built'):
            mod.built = True
        mod.relative_name = name
        mod.batch_index = 0
        mod.filter_index = 1

        mod.in_sequence = False
        mod.uuid = uuid.uuid4().node
        prefix = mod.__class__.__name__
        mod.default_name = camel2snake(prefix) + '_' + str(get_uid(camel2snake(prefix)))

        mod._input_shape = None
        mod._output_shape = None
        mod.keep_output = False
        mod.register_buffer('_output_tensor', None, persistent=False)

        mod._device = get_device()
        #mod._signature = inspect.signature(mod.forward)
        mod.dump_patches = True
        #
        # def getsignature(mod):
        #     return mod._signature
        #
        # def setsignature(mod, value):
        #     mod._signature = value
        #
        # def delsignature(mod):
        #     del mod._signature
        #
        # mod.signature= property(getsignature, setsignature, delsignature, "signature")

        if not hasattr(mod, 'get_root'):
            setattr(mod, 'get_root', MethodType(get_root, mod))

        if not hasattr(mod, 'weights'):
            cls = type(mod)
            cls.weights = property(lambda self: list(self.parameters()))
        if hasattr(mod, 'dims'):
            mod.axis = mod.dims
        if hasattr(module, 'keepdim'):
            value = getattr(module, 'keepdim')
            setattr(module, 'keepdims', value)

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
    # if module.signature is not None:
    #     input_tensor = [to_tensor(module.signature.inputs.value_list[k].get_dummy_tensor()) for k in  range(module.signature.inputs.value_list) if not module.signature.inputs.value_list[k].optional or k==0]
    #     # make a forward pass
    #     # print(x.shape)
    #     module(*input_tensor)
    #
    # # remove these hooks
    # for h in hooks:
    #     h.remove()
    return module
