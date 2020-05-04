from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import inspect
import logging
import operator
import os
import random
import re
import gc
import sys
import uuid
import warnings
from collections import defaultdict
from collections import deque
from copy import copy, deepcopy
from functools import partial, wraps, update_wrapper
from itertools import islice

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import six
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.onnx
import torchvision
from torch._six import container_abcs

from trident.backend.common import to_list, addindent, camel2snake, snake2camel, unpack_singleton, enforce_singleton, OrderedDict, get_signature,get_session,set_session
from trident.backend.pytorch_ops import *

__all__ = ['get_device','set_device','print_network','plot_tensor_grid','summary','Layer', 'Sequential','ModuleList', 'Input', 'get_device', 'load','Combine','ReplayBuffer','try_map_args_and_call']

version=torch.__version__
sys.stderr.write('Pytorch version:{0}.\n'.format(version))
if version<'1.2.0':
    raise ValueError('Not support Pytorch below 1.2' )


def get_device():
    return get_session().device


def set_device(device='cpu'):
    device=device.lower().replace('gpu','cuda')
    if device=='cuda' and not torch.cuda.is_available():
        raise ValueError('Gpu is not available...')
    try:
        set_session('device',device)
        if device=='cpu':
            gcitems = gc.get_objects()
            for i in range(len(gcitems)):
                obj = gcitems[i]
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        obj.to(device)
                    elif isinstance(obj, nn.Module):
                        obj.to(device)
                except Exception:
                    pass
    except Exception as e:
        print(e)


_device =get_device()
if _device is None:
    set_device("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available() and get_device()=='cuda' :
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True




def load(path):
    item=torch.load(path)
    if isinstance(item,nn.Module):
        item.to(_device)
    return item

def save(obj,path,is_compressed=False):
    torch.save(obj,path,_use_new_zipfile_serialization=is_compressed)
    return True



import sys

from functools import partial
from typing import List, IO, Union, Tuple, Type, Callable


def reset_name(module:nn.Module, prefix_dict=None):
    def get_uid(prefix,seq):
        if prefix not in module._uid_prefixs or seq<module._uid_prefixs[prefix]:
            module._uid_prefixs[prefix]=seq
        return module._uid_prefixs[prefix]
    if not hasattr(module,'_uid_prefixs') or prefix_dict is not None:
        module._uid_prefixs=prefix_dict
    if not hasattr(module,'_default_name'):
        module._default_name = camel2snake(module.__class__.__name__) + '_' + str(get_global_uid(camel2snake(module.__class__.__name__)))
    prefix,seq=module._default_name.rsplit('_', 1) #if '_' in module._default_name else
    seq=int(seq)
    module.default_name = prefix + '_' + str(seq-get_uid(prefix,seq)+1)


_UID_PREFIX = defaultdict(int)


def get_global_uid(prefix=''):
    _UID_PREFIX[prefix] += 1
    return _UID_PREFIX[prefix]

class Layer(nn.Module):
    """
    Trident extened nn.Module
    """

    def __init__(self,**kwargs):
        super(Layer, self).__init__()
        self.training = True
        self._built= False
        self._uid_prefixs ={}
        self.rank= kwargs.get('rank',None)
        self._input_shape = None
        self._output_shape = None
        self._output_tensor =None
        self.keep_output= kwargs.get('keep_output',False)
        self.signature=None

        prefix = self.__class__.__name__
        self._default_name= camel2snake(prefix) + '_' + str(get_global_uid(camel2snake(prefix)))
        reset_name(self,self._uid_prefixs)
        self.relative_name=''
        # self._input_shape=None
        # self._output_shape = None
        self.input_filters =None

        self._name = kwargs.get('name')
        #self.dump_patches = True

        self.uuid=uuid.uuid4().node

        self._nodes = None




    @property
    def name(self):
        return self._name if self._name is not None and len(self._name)>0 else self.default_name

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

    def add_module(self, name, module):
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, (nn.Module,Layer)) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                torch.typename(module)))
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("module name should be a string. Got {}".format(torch.typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")

        self._modules[name] = module
        if isinstance(module,Layer):
            reset_name(module, self._uid_prefixs)
            module.relative_name = name if module.relative_name == '' else name + '.' + module.relative_name
            for mod in module.modules():
                if isinstance(mod,Layer) and mod.uuid!=module.uuid:
                    mod.nodes =self.nodes
                    reset_name(mod,self._uid_prefixs)
                    mod.relative_name = name if mod.relative_name == '' else name + '.' + mod.relative_name

        self.nodes=OrderedDict([(mod.uuid,mod)  for mod in list(self.modules()) if isinstance(mod,Layer)])
        for mod in self.modules():
            if isinstance(mod,Layer):
                mod.nodes =self.nodes

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



    @property
    def trainable_weights(self) -> List[nn.Parameter]:
        r"""The list of trainable variables (parameters) of the module.
        Parameters of this module and all its submodules are included.
        .. note::
            The list returned may contain duplicate parameters (e.g. output
            layer shares parameters with embeddings). For most usages, it's not
            necessary to ensure uniqueness.
        """
        return [x for x in self.parameters() if x.requires_grad]

    @property
    def non_trainable_weights(self) -> List[nn.Parameter]:
        r"""The list of trainable variables (parameters) of the module.
        Parameters of this module and all its submodules are included.
        .. note::
            The list returned may contain duplicate parameters (e.g. output
            layer shares parameters with embeddings). For most usages, it's not
            necessary to ensure uniqueness.
        """
        return [x for x in self.parameters() if x.requires_grad==False]

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
        if len(self.weights)==0:
            return False
        elif len(self.weights)>0:
            for k,v in self._parameters.items():
                if v is not None and v.requires_grad==False:
                    return False
            else:
                return True

    @trainable.setter
    def trainable(self,value:bool):
        n=0
        for name, para in self.named_parameters():
            para.requires_grad = value
            n+=np.prod(list(para.size()))
        if value:
            print('{0} parameters have set trainable'.format(n))
        else:
            print('{0} parameters have set untrainable'.format(n))


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
        return  get_device()

    def cuda(self, device=None):
        set_device('cuda')
        super().cuda(device=device)

    def cpu(self):
        super().cpu()
        set_device('cpu')


    @property
    def built(self):
        return self._built

    @property
    def input_shape(self):
        return self._input_shape
    #
    # @property
    # def input_spec(self):
    #     return self._input_spec
    #
    # @input_spec.setter
    # def input_spec(self, value):
    #     for v in nest.flatten(value):
    #         if v is not None and not isinstance(v, InputSpec):
    #             raise TypeError('Layer input_spec must be an instance of InputSpec. '
    #                             'Got: {}'.format(v))
    #     self._input_spec = value

    # @property
    # def input(self):
    #     """Retrieves the input tensor(s) of a layer.
    #     Only applicable if the layer has exactly one input,
    #     i.e. if it is connected to one incoming layer.
    #     Returns:
    #         Input tensor or list of input tensors.
    #     Raises:
    #       RuntimeError: If called in Eager mode.
    #       AttributeError: If no inbound nodes are found.
    #     """
    #     return self._input_tensors





    @input_shape.setter
    def input_shape(self, value):
        if isinstance(value, (list,tuple)) and len(value)>0:
            if isinstance(value[0], torch.Size):
                value=to_tensor(to_numpy(list(value[0]))).int()
            elif isinstance(value[0], torch.Tensor):
                value = value[0].int()
            else:
                value=to_tensor(to_numpy(list(value)))
        elif isinstance(value, int):
            value=to_tensor(to_numpy([value]))
        elif isinstance(value, torch.Size):
            value=to_tensor(to_numpy(list(value))).int()
        elif isinstance(value, torch.Tensor):
            value=value.int()
        elif isinstance(value, np.ndarray) and value.ndim <= 1:
            value=torch.tensor(value.astype(np.uint8))
        else:
            raise ValueError('not valid input_shape')


        if self._built == False :
            self._input_shape =value
            if self._input_shape.ndim == 0:
                self.input_filters = -1
            elif len(self._input_shape) == 1:
                self.input_filters = self._input_shape.item()
            else:
                self.input_filters =int(self._input_shape[0])


            self.build(self._input_shape)
            self._built = True


        elif self._input_shape is not None and self._input_shape.tolist()==to_list(value):
            'input_shape is already assigned, and shape is the same.'
            pass

    @property
    def output_shape(self):
        return  to_tensor(to_numpy(list(self._output_tensor.size()[1:]))) if self._output_shape is None else self._output_shape

    @output_shape.setter
    def output_shape(self, value):
        if isinstance(value, (list, tuple)) and len(value) > 0:
            if isinstance(value[0], torch.Size):
                value = to_tensor(to_numpy(list(value[0]))).int()
            elif isinstance(value[0], torch.Tensor):
                value = value[0].int()
            else:
                value = to_tensor(to_numpy(list(value)))
        elif isinstance(value, int):
            value=to_tensor(to_numpy([value]))
        elif isinstance(value, torch.Size):
            value = to_tensor(to_numpy(list(value))).int()
        elif isinstance(value, torch.Tensor):
            value = value.int()
        elif isinstance(value, np.ndarray) and value.ndim <= 1:
            value = torch.tensor(value.astype(np.uint8))
        else:
            self._output_shape = value
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


    def save_onnx(self,file_path=''):
        input_shape=self.input_shape.copy()
        input_shape.insert(0,1)
        x = torch.randn(*input_shape, requires_grad=True)
        torch_out = self(x)

        # Export the model
        torch.onnx.export(self,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          file_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                        'output': {0: 'batch_size'}})

    def __call__(self, *input, **kwargs):
        for hook in self._forward_pre_hooks.values():
            input = hook(self, input)
        if self._built==False :
            inp= unpack_singleton(input)
            if isinstance(inp, (tuple, list)):
                self.input_shape=inp[0].size()[1:]
            elif isinstance(inp,torch.Tensor):
                self.input_shape = inp.size()[1:]
            else:
                print('input shou be tensor or tuple of tensor')
                self.input_shape=torch.tensor(-1)
            self.build(self.input_shape)
        if torch._C._get_tracing_state():
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(*input, **kwargs)
            result=unpack_singleton(result)

            if hasattr(self,'keep_output') and self.keep_output==True:
                self._output_tensor=result
            if isinstance(result,torch.Tensor) :
                if self._output_shape is None:
                    self.output_shape=result.size()[1:]


        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                result = hook_result
        if len(self._backward_hooks) > 0:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = partial(hook, self)
                    update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
        return result

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
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, nn.Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
                reset_name(value,self._uid_prefixs)
                value.relative_name= name if not hasattr(value,'relative_name') or value.relative_name == '' else name + '.' + value.relative_name
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch.typename(value), name))
                modules[name] = value
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





class Input(Layer):
    def __init__(self, input_shape: (list, tuple,int) = None,name=''):
        super().__init__()
        self.name=name
        if isinstance(input_shape,int):
            input_shape=input_shape,
        self.input_shape=tuple(input_shape)
        self._built=True


    def forward(self, x):
        if x is None:
            return torch.rand(2,*self.input_shape)
        else:
            return x


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

    def __init__(self, *args,name=''):
        super(Sequential, self).__init__(name=name)
        self._built = False
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                module.name=key
                self.add_module(key, module)
        elif len(args) == 1 and isinstance(args[0], (list,nn.ModuleList)):
            for  idx, module in enumerate(args[0]):
                self.add_module(str(idx), module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        self.to(self.device)

    def build(self, input_shape):
        if self._built==False and len(self._modules)>0:
            self.__getitem__(0).input_shape=self.input_shape
            self._built=True

    # def add_module(self, name, module):
    #     r"""Adds a child module to the current module.
    #
    #     The module can be accessed as an attribute using the given name.
    #
    #     Args:
    #         name (string): name of the child module. The child module can be
    #             accessed from this module using the given name
    #         module (Module): child module to be added to the module.
    #     """
    #     if isinstance(module, (list,tuple)):
    #         for i in range(len(module)):
    #             m=module[i]
    #             if isinstance(module, Layer):
    #                 self.add_module(m.name,m)
    #     else:
    #         if not isinstance(module, Layer) and module is not None:
    #             raise TypeError("{} is not a Module subclass".format(torch.typename(module)))
    #         elif not isinstance(name, torch._six.string_classes):
    #             raise TypeError("module name should be a string. Got {}".format(torch.typename(name)))
    #         elif hasattr(self, name) and name not in self._modules:
    #             raise KeyError("attribute '{}' already exists".format(name))
    #         elif '.' in name:
    #             raise KeyError("module name can't contain \".\"")
    #         elif name == '':
    #             raise KeyError("module name can't be empty string \"\"")
    #         if len(self._modules) > 0 and self._input_shape is not None and self[-1].built and self[-1]._output_shape is not None:
    #             module.input_shape = self[-1]._output_shape
    #         self._modules[name] = module
    #         self._output_shape = module._output_shape

    def sync_build(self):
        input_shape=None
        if self[:1] is Input:
            input_shape=self[:1].input_shape
        if input_shape is not None:
            input_shape = list(input_shape)
            input_shape.insert(0, 2)
            data = torch.tensor(np.random.standard_normal(input_shape))
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
            returnDict=OrderedDict()
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



class ModuleList(Layer):
    r"""Holds submodules in a list.

    :class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    Arguments:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if isinstance(modules,dict):
            for key, value in modules.items():
                self.add_module(key, value)
        elif isinstance(modules,(list,tuple)):
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


    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

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


    def __len__(self):
        return len(self._modules)


    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)


    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index, module):
        r"""Insert a given module before a given index in the list.

        Arguments:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self, module):
        r"""Appends a given module to the end of the list.

        Arguments:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        r"""Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self


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

    def __init__(self, *args):
        super(Combine, self).__init__()
        self._built = False

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        elif len(args) == 1 and isinstance(args[0], (list,nn.ModuleList)):
            for  idx, module in enumerate(args[0]):
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
        outputs=[]
        for module in self._modules.values():
            outputs.append(module(*x))
        return tuple(outputs)


class ReplayBuffer:
    def __init__(self, max_size=1000):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        keep_idx=random.choice(range(data.data.size(0)))
        for i in range(data.data.size(0)):
            element=data.data[i]
            element = torch.unsqueeze(element, 0)

            if len(self.data) > 10 and random.uniform(0, 1) > 0.7:
                i = random.randint(0, self.max_size - 1)
                to_return.append(self.data[i].clone())
            else:
                to_return.append(element)

            if 0<len(self.data) < self.max_size and keep_idx==i:
                self.data.append(element)
            elif len(self.data) == self.max_size and random.randint(0,10)%3==0 and keep_idx==i:
                self.data[random.randint(0, self.max_size - 1)] = element

        to_return=shuffle(torch.cat(to_return))
        return to_return

    def push_only(self, data):
        element=random_choice(data)
        element = torch.unsqueeze(element, 0)
        if len(self.data) < self.max_size:
            self.data.append(element)


def print_network(net, verbose=False):
    num_params = 0
    for i, param in enumerate(net.parameters()):
        num_params += param.numel()
    if verbose:
        logging.info(net)
    logging.info('Total number of parameters: %d\n' % num_params)




def plot_tensor_grid(batch_tensor, save_filename=None):
    ''' Helper to visualize a batch of images.
        A non-None filename saves instead of doing a show()'''

    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0))
    if save_filename is not None:
        torchvision.utils.save_image(batch_tensor, save_filename, padding=5)
    else:
        plt.show()




def summary(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key =module.name
            summary[m_key] = OrderedDict()
            summary[m_key]["keep_output"] = module.keep_output
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            summary[m_key]["flops"]=np.array([0],dtype=np.float64)
            summary[m_key]["macc"] = np.array([0], dtype=np.float64)
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.shape)))
                summary[m_key]["weight"] =list(module.weight.shape)
                summary[m_key]["trainable"] = module.weight.requires_grad
                summary[m_key]["flops"] += (2*np.prod(np.array(summary[m_key]["weight"]).astype(np.float64))-1) * np.prod(np.array(summary[m_key]["output_shape"][2:]).astype(np.float64))
                summary[m_key]["macc"] += np.prod(np.array(summary[m_key]["weight"]).astype(np.float64)) * np.prod(np.array(summary[m_key]["output_shape"][2:]).astype(np.float64))

            if hasattr(module, "bias") and module.bias is not None and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.shape)))
                summary[m_key]["bias"] =list(module.bias.shape)
                summary[m_key]["flops"]+=np.prod(np.array(summary[m_key]["bias"]).astype(np.float64))*np.prod(np.array( summary[m_key]["output_shape"][2:]).astype(np.float64))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, (nn.Sequential,Sequential,nn.ModuleList,ModuleList))
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    if isinstance(input_size, int):
        x = [torch.rand(1, input_size).type(dtype).to("cuda" if model.weights[0].data.is_cuda else "cpu")]
    else:
        # batch_size of 2 for batchnorm
        x = [torch.rand(1, *in_size).type(dtype).to("cuda" if model.weights[0].data.is_cuda else "cpu") for in_size in input_size]
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
        is_keep = '★' if summary[layer]["keep_output"] else ''
        line_new = "{0:<40s} {1:<20s}  {2:<20s} {3:<8s}  {4:<8}  {5:<12}".format(
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
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = np.abs(np.prod(np.array(list(input_size))) * batch_size * 4. / (1024 ** 2.))
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









def summary_str(model):
    """ Get a string representation of model building blocks and parameter counts. """
    indent_list, name_list, count_list = [], [], []
    def module_info(m, name, indent_level):
        count_list.append(sum([np.prod(list(p.size())) for p in m.parameters()]))
        indent_list.append(indent_level)
        name_list.append(name)
        for name, child in m.named_children():
            if name.isdigit():
                name = child._get_name()
            module_info(child, name, indent_level+1)
    module_info(model, model._get_name(), 0)
    max_indent = max(indent_list)*4
    max_name = max(len(x) for x in name_list)+max_indent+2
    max_param = len(str(count_list[0]))+max_name+2
    out = ['Blocks{:>{w}}'.format('Params', w=max_param-6)]
    out += ['-'*max_param]
    for indent, name, param in zip(indent_list, name_list, count_list):
        s0 = '    '*indent
        s1 = '{:{w}}'.format(name, w=max_name-len(s0))
        s2 = '{:>{w}}'.format(param, w=max_param-len(s1)-len(s0))
        out += [s0+s1+s2]
    return '\n'.join(out)



import gc
import subprocess

import numpy as np
import pandas as pd
import torch


class ModelSummary(object):

    def __init__(self, model, mode='full'):
        '''
        Generates summaries of model layers and dimensions.
        '''
        self.model = model
        self.mode = mode
        self.in_sizes = []
        self.out_sizes = []

        self.summarize()

    def __str__(self):
        return self.summary.__str__()

    def __repr__(self):
        return self.summary.__str__()

    def named_modules(self):
        if self.mode == 'full':
            mods = self.model.named_modules()
            mods = list(mods)[1:]  # do not include root module (LightningModule)
        elif self.mode == 'top':
            # the children are the top-level modules
            mods = self.model.named_children()
        else:
            mods = []
        return list(mods)

    def get_variable_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        mods = self.named_modules()
        in_sizes = []
        out_sizes = []
        input_ = self.model.example_input_array

        if self.model.on_gpu:
            input_ = input_.cuda()

        if self.model.trainer.use_amp:
            input_ = input_.half()

        with torch.no_grad():

            for _, m in mods:
                if type(input_) is list or type(input_) is tuple:  # pragma: no cover
                    out = m(*input_)
                else:
                    out = m(input_)

                if type(input_) is tuple or type(input_) is list:  # pragma: no cover
                    in_size = []
                    for x in input_:
                        if type(x) is list:
                            in_size.append(len(x))
                        else:
                            in_size.append(x.size())
                else:
                    in_size = np.array(input_.size())

                in_sizes.append(in_size)

                if type(out) is tuple or type(out) is list:  # pragma: no cover
                    out_size = np.asarray([x.size() for x in out])
                else:
                    out_size = np.array(out.size())

                out_sizes.append(out_size)
                input_ = out

        self.in_sizes = in_sizes
        self.out_sizes = out_sizes
        assert len(in_sizes) == len(out_sizes)
        return

    def get_layer_names(self):
        '''Collect Layer Names'''
        mods = self.named_modules()
        names = []
        layers = []
        for name, m in mods:
            names += [name]
            layers += [str(m.__class__)]

        layer_types = [x.split('.')[-1][:-2] for x in layers]

        self.layer_names = names
        self.layer_types = layer_types
        return

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        mods = self.named_modules()
        sizes = []
        for _, m in mods:
            p = list(m.parameters())
            modsz = []
            for j in range(len(p)):
                modsz.append(np.array(p[j].size()))
            sizes.append(modsz)

        self.param_sizes = sizes
        return

    def get_parameter_nums(self):
        '''Get number of parameters in each layer'''
        param_nums = []
        for mod in self.param_sizes:
            all_params = 0
            for p in mod:
                all_params += np.prod(p)
            param_nums.append(all_params)
        self.param_nums = param_nums
        return

    def make_summary(self):
        '''
        Makes a summary listing with:

        Layer Name, Layer Type, Input Size, Output Size, Number of Parameters
        '''

        cols = ['Name', 'Type', 'Params']
        if self.model.example_input_array is not None:
            cols.extend(['In_sizes', 'Out_sizes'])

        df = pd.DataFrame(np.zeros((len(self.layer_names), len(cols))))
        df.columns = cols

        df['Name'] = self.layer_names
        df['Type'] = self.layer_types
        df['Params'] = self.param_nums
        df['Params'] = df['Params'].map(get_human_readable_count)

        if self.model.example_input_array is not None:
            df['In_sizes'] = self.in_sizes
            df['Out_sizes'] = self.out_sizes

        self.summary = df
        return

    def summarize(self):
        self.get_layer_names()
        self.get_parameter_sizes()
        self.get_parameter_nums()

        if self.model.example_input_array is not None:
            self.get_variable_sizes()
        self.make_summary()


def print_mem_stack():  # pragma: no cover
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


def get_memory_profile(mode):
    """
    'all' means return memory for all gpus
    'min_max' means return memory for max and min
    :param mode:
    :return:
    """
    memory_map = get_gpu_memory_map()

    if mode == 'min_max':
        min_mem = 1000000
        min_k = None
        max_mem = 0
        max_k = None
        for k, v in memory_map:
            if v > max_mem:
                max_mem = v
                max_k = k
            if v < min_mem:
                min_mem = v
                min_k = k

        memory_map = {min_k: min_mem, max_k: max_mem}

    return memory_map


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = {}
    for k, v in zip(range(len(gpu_memory)), gpu_memory):
        k = 'gpu_{k}'
        gpu_memory_map[k] = v
    return gpu_memory_map


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
    return '{int(number):,d} {labels[index]}'


def try_map_args_and_call(fn, data: OrderedDict,data_feed=None,):
    if isinstance(fn,torch.Tensor):
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
            out=fn(*arg_map.value_list)
            return out
        elif hasattr(fn,'signature') and callable(fn):
            for arg in fn.signature.key_list:
                if arg in data:
                    arg_map[arg]=data[arg]
                elif arg in data_feed:
                    arg_map[arg]=data[data_feed[arg]]
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








