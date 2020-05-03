import copy
import inspect
import itertools
import random
import sys
import uuid
from collections import defaultdict,namedtuple
from functools import partial, wraps, update_wrapper
from itertools import islice
from typing import List
import weakref
import numpy as np
import tensorflow as tf
import tensorflow.python.eager as tfe
from tensorflow.python import enable_eager_execution

from tensorflow.python.module import module
from tensorflow.python.util import object_identity
from ..data.utils import pickle_it,unpickle
from .common import floatx, addindent, OrderedDict, camel2snake, get_signature, get_time_suffix, format_time, \
    get_terminal_size, snake2camel, PrintException, to_list, unpack_singleton, enforce_singleton, OrderedDict, \
    get_signature,normalize_padding
from .tensorflow_ops import *

__all__ = [ 'Layer', 'get_flops','Sequential',
           'ConcatContainer', 'ReplayBuffer','summary']
gpus = tf.config.list_physical_devices('GPU')
def get_device():

    if gpus:
        return gpus[0]
    else:
        return "/cpu:0"


version = tf.version
sys.stdout.write('Tensorflow version:{0}.\n'.format(version.VERSION))

if version.VERSION < '2.0.0':
    raise ValueError('Not support Tensorflow below 2.0\n')

sys.stdout.write('use device:{0}.\n'.format(get_device()))

enable_eager_execution()
tf.executing_eagerly()
sys.stdout.write('executing_eagerly\n')



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






def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph, run_meta=run_meta,
                                          cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


def _is_not_trainable_variable(obj):
  return module._is_variable(obj) and not getattr(obj, "trainable", False)

_UID_PREFIXES = defaultdict(int)

def get_uid(prefix=''):
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]

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


class Layer(tf.Module):
    """
    Trident extened nn.Module
    """
    _version = 1
    def __init__(self,name=None,**kwargs):
        super(Layer, self).__init__()
        self.training=True
        self._built= False
        self._modules = OrderedDict()
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._state_dict_hooks = OrderedDict()
        self._load_state_dict_pre_hooks = OrderedDict()
        self._input_shape = None
        self._output_shape = None


        self._output_tensor =None
        self.keep_output=kwargs.get('keep_output',False)
        self.signature=None

        self.input_filters =None
        self.uuid=uuid.uuid4().node
        prefix = self.__class__.__name__
        self.defaultname = camel2snake(prefix) + '_' + str(get_uid(prefix))
        self._name = kwargs.get('name',name)
        # self.dump_patches = True

        self.uuid = uuid.uuid4().node

        self._nodes = None

        self._device='/cpu:0' if gpus is None or len(gpus)==0 else gpus[0]


    @property
    def name(self):
        return self._name if self._name is not None and len(self._name)>0 else self.defaultname

    @name.setter
    def name(self,value):
        self._name=value


    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self,value):
        if self._nodes!=value:
            self._nodes=value
            for mod in self.modules():
                mod._nodes=value
    # @property
    # def nodes(self):
    #     return self._nodes
    #
    # @nodes.setter
    # def nodes(self,value):
    #     if self._nodes!=value:
    #         self._nodes=value
    #         for mod in self.modules():
    #             mod._nodes=value

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
        handle =RemovableHandle(self._forward_pre_hooks)
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
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle =RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle

    def _get_name(self):
        return self.__class__.__name__

    def register_buffer(self, name, tensor):
        r"""Adds a persistent buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the persistent state.

        Buffers can be accessed as attributes using given names.

        Args:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor): buffer to be registered.

        Example::

            >>> self.register_buffer('running_mean', tf.zeros([5]))

        """
        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("buffer name should be a string. "
                            "Got {}".format(type(name).__name__))
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor,tf.Tensor) and not is_tensor(tensor):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(torch Tensor or None required)"
                            .format(type(tensor).__name__, name))
        else:
            self._buffers[name] = tensor

    def register_parameter(self, name, param):
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter): parameter to be added to the module.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

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
                            "(tf.Variable or None required)"
                            .format(type(param).__name__, name))
        else:

            self._parameters[name] = param

    def add_module(self, name, module):
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if name is  None or len(name)==0:
            name=module._name

        if module is None :
            raise KeyError("module  can't be None")
        elif not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(type(name).__name__))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        elif isinstance(module, Layer) :
            # nodes = self._nodes.copy()
            # for k, v in module.nodes.items():
            #     nodes[k] = v
            self._modules[name] = module

            self.nodes = OrderedDict([(mod.uuid, mod) for mod in list(self.modules()) if isinstance(mod, Layer)])
            for mod in self.modules():
                if isinstance(mod, Layer):
                    mod.nodes = self.nodes

        elif inspect.isfunction(module) or callable(module):
            module.__name__=name
            self._modules[name] = module

        else:
            raise  ValueError('Not valid module')


    def add(self, module):
        if module is None :
            raise KeyError("module  can't be None")
        elif isinstance(module, Layer) :
            # nodes = self._nodes.copy()
            # for k, v in module.nodes.items():
            #     nodes[k] = v
            self.add_module(str(len(self._modules)),module)
            # self.nodes = nodes
            # for mod in self.modules():
            #     mod.nodes = nodes

        else:
            raise  ValueError('Not valid module')

    def build(self, input_shape):
        pass  #pass if no need shape infer

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.
        Assumes that the layer will be built
        to match that input shape provided.
        # Arguments
            input_shape: Shape tuple (tuple of integers)
                or list of shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.
        # Returns
            An output shape tuple.
        """
        if not self._built:
            self.input_shape=input_shape
        return self.output_shape

    def cuda(self, device=None):
        r"""Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        self._device='/cpu:0'

    def cpu(self):
        r"""Moves all model parameters and buffers to the CPU.

        Returns:
            Module: self
        """
        self._device='/cpu:0'
        tf.device(self._device)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)


        for key, param in self._parameters.items():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't want to
                # track autograd history of `param_applied`, so we have to use
                # `with torch.no_grad():`
                #with torch.no_grad():
                param_applied = fn(param)
                self._parameters[key] =  tf.Variable(param_applied, trainable=param.trainable)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def apply(self, fn):
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




    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable}

        if hasattr(self, 'batch_input_shape'):
            config['batch_input_shape'] = self.batch_input_shape

        if hasattr(self, 'dtype'):
            config['dtype'] = self.dtype

        return config

    @property
    def device(self):
        return tf.device()

    @property
    def built(self):
        return self._built

    @property
    def input_shape(self):
        return self._input_shape


    @input_shape.setter
    def input_shape(self, value):
        if isinstance(value, (list,tuple)) and len(value)>0:
            value =tf.TensorShape(list(value))
        elif is_tensor(value):
            value=to_numpy(value).tolist()
        elif isinstance(value,int):#
            value =tf.TensorShape(value)
        elif isinstance(value, np.ndarray) and value.ndim <= 1:
            value=tf.TensorShape(value.astype(np.uint8).tolist())
        elif isinstance(value,tf.TensorShape):
            value = value
        else:
            raise ValueError('not valid input_shape')


        if self._built == False :
            self._input_shape =value
            if len(self._input_shape) == 0:
                self.input_filters = -1
            else:
                self.input_filters =int(self._input_shape[-1])


            self.build(self._input_shape)
            self._built = True


        elif self._input_shape is not None and self._input_shape==tuple(to_list(value)):
            'input_shape is already assigned, and shape is the same.'
            pass

    @property
    def output_shape(self):
        return  self._output_shape

    @output_shape.setter
    def output_shape(self, value):
        if isinstance(value, (list, tuple)) and len(value) > 0:
            value = tf.TensorShape(list(value))
        elif is_tensor(value):
            value = tf.TensorShape(to_numpy(value).tolist())
        elif isinstance(value,int):#
            value =tf.TensorShape(value)
        elif isinstance(value, np.ndarray) and value.ndim <= 1:
            value = tf.TensorShape(value.astype(np.uint8).tolist())
        elif isinstance(value,tf.TensorShape):
            value = value
        else:
            raise ValueError('not valid input_shape')
        self._output_shape = value

    @property
    def input(self):
        return  NotImplemented

    @property
    def output(self):
        """Retrieves the output tensor(s) of a layer.
        for memory saving issue, we don'tb prefer to keep every input/output
        tensor in every layer.You should set self.keep.output flag to True, and then
        retrive the output tensor when the calll() is executing.
        # Returns
            Output tensor or list of output tensors.
        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        if self.keep_output==False:
            raise ValueError('Layer {0} has not set self.keep.output  to True, cannot access output '.format(self.name))
        return list(self._output_tensor) if isinstance(self._output_tensor,tuple) else self._output_tensor

    def reset_parameters(self):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`Layer.state_dict`.
        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.
        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else tf.stop_gradient(param)
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf if keep_vars else tf.stop_gradient(buf)

    def  state_dict(self, destination=None, prefix='', keep_vars=False):
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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.
        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
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
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
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
                                      'the shape in current model is {}.'.format(key, input_param.shape, param.shape))
                    continue

                try:
                    setattr(self, name,tf.Variable(to_numpy(input_param)))
                    #param=input_param

                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occured : {}.'.format(key, param.shape, input_param.shape,
                                                                          ex.args))
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
        by this module's :meth:`~torch.nn.Module.state_dict` function.
        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
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
                error_msgs.insert(0,
                    'Missing key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'.format(self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def save(self, file_path=''):
        #save({'state_dict': self.state_dict()}, file_path)
        pickle_it(file_path,{'state_dict': self.state_dict()})

    def save_onnx(self,file_path=''):
        pass


    def forward(self, *input, **kwargs):
        r"""Defines the computation performed at every call.

        Should be overridden by all subclasses.

        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.
        """
        raise NotImplementedError

    def _slow_forward(self, *input, **kwargs):
        pass
        # tracing_state = torch._C._get_tracing_state()
        # if not tracing_state or isinstance(self.forward, torch._C.ScriptMethod):
        #     return self.forward(*input, **kwargs)
        # recording_scopes = torch.jit._trace_module_map is not None
        # if recording_scopes:
        #     name = torch.jit._trace_module_map[self] if self in torch.jit._trace_module_map else None
        #     if name:
        #         cur_scope_name = tracing_state.current_scope()
        #         tracing_state.push_scope(name)
        #     else:
        #         recording_scopes = False
        # try:
        #     result = self.forward(*input, **kwargs)
        # finally:
        #     if recording_scopes:
        #         tracing_state.pop_scope()
        # return result


    def __call__(self, *input, **kwargs):
        for hook in self._forward_pre_hooks.values():
            input = hook(self, input)
        if self._built == False:
            inp = unpack_singleton(input)
            if isinstance(inp, (tuple, list)):
                self.input_shape=inp[0].get_shape()[1:]
            elif is_tensor(inp):
                self.input_shape =inp.get_shape()[1:]
            elif isinstance(inp,np.ndarray):
                self.input_shape =tf.TensorShape(inp.shape[1:].tolist())
                inp=to_tensor(inp)
            else:
                print('input shou be tensor or tuple of tensor')
                self.input_shape=tf.TensorShape(None)
            self.build(self.input_shape)

        #don't use result = self.forward(i*nput, **kwargs) because EagerTensor will splited as a tuple....
        try:
            result = self.forward(*input, **kwargs)
            result = unpack_singleton(result)
            if hasattr(self, 'keep_output') and self.keep_output == True:
                self._output_tensor = result
            if is_tensor(result):
                if self._output_shape is None:
                    self.output_shape = result.get_shape()[1:]

            for hook in self._forward_hooks.values():
                hook_result = hook(self, input, result)
                if hook_result is not None:
                    result = hook_result
                    return result
            return result
        except Exception as e:
            print('{0} ({1} call failed.)'.format(self.name,self.defaultname))
            print(e)
            raise e



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
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        if isinstance(value, tf.Variable):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(tf.Variable or None expected)"
                                .format(type(value).__name__, name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Layer):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(type(value).__name__, name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not is_tensor(value):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(type(value).__name__, name))
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
                istensor=is_tensor(v)
                key=v.ref() if istensor else v
                if v is None or key in memo:
                    continue
                name = module_prefix + ('.' if module_prefix else '') + k
                memo.add(key)
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

        Example::

            >>> for param in model.parameters():
            >>>     print(type(param.data), param.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix='', recurse=True):
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

        Example::

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

        Example::

            >>> for name, buf in self.named_buffers():
            >>>    if name in ['running_var']:
            >>>        print(buf.size())

        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse)
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

        Example::

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

        Example::

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

        Example::

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

    def train(self, mode=True):
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
        for module in self.modules():
            module.training = mode
        return self

    def eval(self):
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
            p.data = w.data


    @property
    def trainable(self):
        if len(self.weights)==0:
            return False
        elif len(self.weights)>0:
            for k,v in self._parameters.items():
                if v is not None and v.trainable==False:
                    return False
            else:
                return True

    @trainable.setter
    def trainable(self,value:bool):
        n=0
        need_update=False
        for name, para in self.named_parameters():
            if para.trainable!=value:
                para.trainable = value
                n+=np.prod(list(para.size()))
                need_update=True
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

    def __init__(self, *args,name=None):
        super(Sequential, self).__init__(name=name)
        self._built = False
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                module.name = key
                self.add_module(key, module)
        elif len(args) == 1 and isinstance(args[0], list):
            for idx, module in enumerate(args[0]):
                self.add_module(str(idx), module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

        # self.to(self.device)

    def build(self, input_shape):
        if self._built==False and len(self._modules)>0:
            self.__getitem__(0).input_shape=self.input_shape
            self._built=True


    def sync_build(self):
        input_shape=None
        # if self[:1] is Input:
        #     input_shape=self[:1].input_shape
        if input_shape is not None:
            input_shape = list(input_shape)
            input_shape.insert(0, 2)
            data = to_tensor(np.random.standard_normal(input_shape))
            out = self(data)

    def remove_at(self,idx):
        self.__delitem__(idx)


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
            returnDict = OrderedDict()
            for k, v in list(self._modules.items())[idx]:
                returnDict[k] = v
            return returnDict
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
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys


    def forward(self, *x):
        x = enforce_singleton(x)
        for module in self._modules.values():
            x = module(x)
        return x


class Combine(tf.keras.Model):
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

    def __init__(self, *args):
        super(Combine, self).__init__()
        self._built = False

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                module.__name__ = key
                self._layers.append(module)
        elif len(args) == 1 and isinstance(args[0], (list)):
            for idx, module in enumerate(args[0]):
                self._layers.append(module)
        else:
            for idx, module in enumerate(args):
                self._layers.append(module)

    @property
    def layers(self):
        return self._layers

    def __len__(self):
        return len(self._layers)

    def __dir__(self):
        keys = super(Combine, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def call(self, inputs, training=None, mask=None):  # pylint: disable=redefined-outer-name
        if self._is_graph_network:
            if not self.built:
                self._init_graph_network(self.inputs, self.outputs, name=self.name)
            return super(Combine, self).call(inputs, training=training, mask=mask)

        outputs = []
        for layer in self._layers:
            outputs.append(layer(inputs))
        return tuple(outputs)


def compute_output_shape(self, input_shape):
    shape = input_shape
    for layer in self.layers:
        shape = layer.compute_output_shape(shape)
    return shape


class ConcatContainer(tf.keras.Model):
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

    def __init__(self, *args, **kwargs):
        super(ConcatContainer, self).__init__()
        self.axis = kwargs.get('axis', -1)
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self._modules[len(self._modules)] = module
        else:
            for idx, module in enumerate(args):
                self._modules[idx] = module
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
        keys = super(ConcatContainer, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def call(self, x, **kwargs):
        results = []
        for module in self._modules.values():
            x1 = module(x)
            results.append(x1)
        return tf.concat(results, axis=-1)


class ReplayBuffer:
    def __init__(self, max_size=1000):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = tf.keras.backend.expand_dims(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return tf.concat(to_return,axis=-1)



def count_params(weights):
    """Count the total number of scalars composing the weights.

    Arguments:
        weights: An iterable containing the weights on which to compute params

    Returns:
        The total number of scalars composing the weights
    """
    return int(sum(np.prod(p.shape.as_list()) for p in object_identity.ObjectIdentitySet(weights)))


def calculate_flops(gen:Layer):
    """
    Calculate the flops given a generator of pytorch model.
    It only compute the flops of forward pass.

    Example:
        >>> net = torchvision.models.resnet18()
        >>> calculate_flops(net.children())
    """
    flops = 0
    mods=gen.named_modules()
    mods=list(mods)[1:]
    param_nums = []
    param_sizes = []
    for mod_name,mod in mods:
        p = list(mod.parameters())
        modsz = []
        all_params = 0
        for j in range(len(p)):
            modsz.append(np.array(p[j].size()))
            all_params += np.prod(p[j].size())

        param_nums.append(all_params)
        param_sizes.append(modsz)

    return np.array(param_nums).sum()


# net = torchvision.models.resnet18()
# flops = calculate_flops(net.children())
# print(flops / 10 ** 9, 'G')  # 11.435429919 G



def summary(model, input_size, batch_size=-1):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key =module.name if hasattr(module,'name') else camel2snake(module.__class__.__name__) + '_' + str(get_uid(module.__class__.__name__))
            summary[m_key] = OrderedDict()
            summary[m_key]["keep_output"]=module.keep_output
            summary[m_key]["input_shape"] = list(to_numpy(input[0]).shape)
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output ]
            else:
                summary[m_key]["output_shape"] = list(to_numpy(output).shape)
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            summary[m_key]["flops"]=np.array([0],dtype=np.float64)
            summary[m_key]["macc"] = np.array([0], dtype=np.float64)
            if hasattr(module, "weight") and hasattr(module.weight, "shape"):
                params += np.prod(np.array(to_numpy(module.weight).shape))
                summary[m_key]["weight"] =list(to_numpy(module.weight).shape)
                summary[m_key]["trainable"] = module.weight.trainable
                summary[m_key]["flops"] += (2*np.prod(np.array(summary[m_key]["weight"]).astype(np.float64))-1) * np.prod(np.array(summary[m_key]["output_shape"][2:]).astype(np.float64))
                summary[m_key]["macc"] += np.prod(np.array(summary[m_key]["weight"]).astype(np.float64)) * np.prod(np.array(summary[m_key]["output_shape"][2:]).astype(np.float64))

            if hasattr(module, "bias") and module.bias is not None and hasattr(module.bias, "shape"):
                params += np.prod(np.array(to_numpy(module.bias).shape))
                summary[m_key]["bias"] =list(to_numpy(module.bias).shape)
                summary[m_key]["flops"]+=np.prod(np.array(summary[m_key]["bias"]).astype(np.float64))*np.prod(np.array( summary[m_key]["output_shape"][2:]).astype(np.float64))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, Sequential)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))


    # multiple inputs to the network

    if isinstance(input_size, tuple):
        input_size = [input_size]

    if isinstance(input_size, int):
        x = [ to_tensor(np.random.standard_normal((1, *input_size)).astype(np.float32))]
    else:
        # batch_size of 2 for batchnorm
        x = [ to_tensor(np.random.standard_normal((1, *in_size)).astype(np.float32)) for in_size in input_size]
    # p    rint(type(x[0]))

    # create properties

    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("--------------------------------------------------------------------------------------------------------------------------------")
    line_new = "{0:^40s} {1:^20s}  {2:^20s} {3:^8s}  {4:^8s}  {5:^12s}".format("Layer (type)", "Output Shape","Weight ","Bias", "Param #", "FLOPS #")
    print(line_new)
    print("==============================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    flops=np.array([0],dtype=np.float64)
    macc=0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        is_keep='★' if summary[layer]["keep_output"] else ''
        line_new = "{0:<40s} {1:<20s}  {2:^20s} {3:^8s}  {4:^8}  {5:^12}".format(
            layer,
            is_keep+str(summary[layer]["output_shape"]),
            str(summary[layer]["weight"] if 'weight' in summary[layer] else ''),
            str(summary[layer]["bias"] if 'bias' in summary[layer] else ''),
            summary[layer]["nb_params"],
            summary[layer]["flops"][0]
        )
        total_params += summary[layer]["nb_params"]
        flops+= float(summary[layer]["flops"])
        macc += float(summary[layer]["macc"][0])
        total_output += np.prod(to_numpy(summary[layer]["output_shape"]))
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = np.abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = np.abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = np.abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("Total MACC: {0:,}".format(round(macc,0)))
    print("Total FLOPs: {0:.5f} GFLOPs".format(np.round(flops / 10.**9, 5)[0]))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary







def try_map_args_and_call(fn, data: OrderedDict,data_feed=None,):
    if isinstance(fn,tf.Tensor) or 'EagerTensor' in fn.__class__.__name__:
        return fn
    else:
        arg_map = OrderedDict()
        if isinstance(fn,Layer) :
            for arg in fn.signature.key_list:
                if arg in data_feed:
                    arg_map[arg]=data[data_feed[arg]]
                else:
                    raise ValueError('arg :{0} cannot mapping correctly!'.format(arg))
            #print('arg_map',arg_map.key_list)
            if len(fn.signature.key_list)==1:
                inp=unpack_singleton(arg_map.value_list)
                out = fn(inp)
                return out
            else:
                out=fn(*arg_map.value_list)
                return out
        elif hasattr(fn,'signature') and callable(fn):
            for arg in fn.signature.key_list:
                if arg in data:
                    arg_map[arg]=data[arg]
                elif arg in data_feed:
                    arg_map[arg]=data[data_feed[arg]]
                elif arg=='y_pred' and  'output' in data:
                    arg_map[arg] = data['output']
                elif arg=='y_true' and  'target' in data:
                    arg_map[arg] = data['target']
                elif arg=='y_true' and  'label' in data:
                    arg_map[arg] = data['label']
                elif arg=='label' and  'target' in data:
                    arg_map[arg] = data['target']
                else:

                    raise ValueError('arg :{0} cannot mapping correctly!'.format(arg))
            #print('arg_map', arg_map.key_list)
            out=fn(*arg_map.value_list)
            return out
        elif  callable(fn):

            args=get_signature(fn).key_list
            for arg in args:
                if arg in  data_feed:
                    arg_map[arg]=data[data_feed[arg]]
                else:
                    arg_map[arg] = ''
            #print('arg_map', arg_map.key_list)
            out = fn(*arg_map.value_list)
            return out
        else:

            print('uncomplete arg_map', arg_map.key_list)




