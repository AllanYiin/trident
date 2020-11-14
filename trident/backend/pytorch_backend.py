from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numbers
import copy
import itertools
import logging
import operator
import random
import sys
import uuid
from collections import defaultdict
from copy import copy
from functools import update_wrapper,partial
from typing import List, Tuple, Optional, Union
from itertools import islice
from distutils.version import Version, LooseVersion
import torch.nn as nn
import torch.onnx
from torch._six import container_abcs
from torch.nn.parameter import Parameter
from  trident.backend.tensorspec import *
from trident.backend.common import to_list, addindent, camel2snake, unpack_singleton, enforce_singleton, OrderedDict, get_session, set_session, get_session_value, \
    PrintException,Signature
from trident.backend.tensorspec import *
from trident.backend.pytorch_ops import *

__all__ = ['get_device', 'set_device', 'Layer', 'Sequential', 'ModuleList', 'print_network', 'summary', 'load', 'save', 'Combine',  'try_map_args_and_call','print_mem_stack',
           'normalize_padding','fix_layer']




version = torch.__version__
sys.stdout.write('Pytorch version:{0}.\n'.format(version))

pt_version = LooseVersion(vstring=version)
base_version = LooseVersion(vstring='1.2.0')
amp_version = LooseVersion(vstring='1.6.0')

if pt_version.version < base_version.version:
    raise ValueError('Not support Pytorch older then version 1.2')
elif pt_version.version >= amp_version.version:
    set_session('amp_available', True if torch.cuda.is_available() and pt_version >= amp_version else False)
    if get_session_value('amp_available') == True:
        sys.stdout.write('Automatic Mixed Precision Support:{0}.\n'.format(True))
    else:
        sys.stdout.write('Automatic Mixed Precision Support:{0}.\n'.format(False))


def get_device():
    """get current device

    Returns: device string ('cpu', 'cuda)

    """
    if get_session().device is None:
        set_device("cuda" if torch.cuda.is_available() else "cpu")
    return get_session().device


def set_device(device='cpu'):
    device = device.lower().replace('gpu', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Gpu is not available...')
    try:
        set_session('device', device)
        if device == 'cpu':
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


if torch.cuda.is_available() and get_device() == 'cuda':
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def load(f):
    """

    Args:
        f: a file-like object (has to implement :meth:`read`, :meth`readline`, :meth`tell`, and :meth`seek`),
            or a string or os.PathLike object containing a file name

    Returns:

    """

    item = torch.load(f)
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
    torch.save(obj, f, _use_new_zipfile_serialization=is_compressed)
    return True





def reset_name(module: nn.Module, prefix_dict=None):
    def get_uid(prefix, seq):
        if prefix not in module._uid_prefixs or seq < module._uid_prefixs[prefix]:
            module._uid_prefixs[prefix] = seq
        return module._uid_prefixs[prefix]

    if not hasattr(module, '_uid_prefixs') or prefix_dict is not None:
        module._uid_prefixs = prefix_dict
    if not hasattr(module, '_default_name'):
        module._default_name = camel2snake(module.__class__.__name__) + '_' + str(get_global_uid(camel2snake(module.__class__.__name__)))
    prefix, seq = module._default_name.rsplit('_', 1)  # if '_' in module._default_name else
    seq = int(seq)
    module.default_name = prefix + '_' + str(seq - get_uid(prefix, seq) + 1)
    module.__name__ = module._name if hasattr(module, '_name') else module.default_name


_UID_PREFIX = defaultdict(int)


def get_global_uid(prefix=''):
    _UID_PREFIX[prefix] += 1
    return _UID_PREFIX[prefix]


class Layer(nn.Module):
    """Trident extened pytorch nn.Module as base layer class.

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




    """

    def __init__(self, name=None, keep_output=False, **kwargs):
        """

        Args:
            name (str) :name of the layer.
            keep_output (bool) :whether need to kept output tensor in execution time.


        """
        super(Layer, self).__init__()
        self.batch_index=0
        self.filter_index = 1
        self.in_sequence=kwargs.get('in_sequence',False)
        if self.in_sequence:
            self.filter_index = -1
        self.training = True
        self._built = False
        self.rank = kwargs.get('rank', None)
        self._non_persistent_buffers_set = set()
        self.uuid = uuid.uuid4().node
        self._nodes = None
        self._uid_prefixs = {}
        self._name =name

        prefix = self.__class__.__name__
        self.default_name = camel2snake(prefix) + '_' + str(get_global_uid(camel2snake(prefix)))
        self.relative_name = ''
        reset_name(self, self._uid_prefixs)

        self._input_shape = None
        self.input_filters = None
        self.input_spec=None
        self._output_shape = None
        self.keep_output = keep_output
        self._output_tensor = None

        self._signature = None

        # self.dump_patches = True

    def forward(self, *input,**kwargs):
        r"""Defines the computation performed at every call.

        Should be overridden by all subclasses.

        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.
        """
        raise NotImplementedError

    @property
    def name(self):
        """If not assign name , it will return the default_name"""
        return self._name if self._name is not None and len(self._name) > 0 else self.relative_name

    @name.setter
    def name(self, value):
        self._name = value
        self.__name__ = value
        self.signature=None

    @property
    def nodes(self):
        """"The whole tree structured OrderedDict { uuid : module } , for module to access any node in this structures, ex. Shortcut"""
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
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("module name should be a string. Got {}".format(torch.typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            #name=name.replace('.','_')
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")

        self._modules[name] = module
        if isinstance(module, Layer):
            reset_name(module, self._uid_prefixs)
            module.relative_name = name if module.relative_name == '' else name + '.' + module.relative_name
            for mod in module.modules():
                if isinstance(mod, Layer) and mod.uuid != module.uuid:
                    mod.nodes = self.nodes
                    reset_name(mod, self._uid_prefixs)
                    mod.relative_name = name if mod.relative_name == '' else name + '.' + mod.relative_name

        self.nodes = OrderedDict([(mod.uuid, mod) for mod in list(self.modules()) if isinstance(mod, Layer)])
        for mod in self.modules():
            if isinstance(mod, Layer):
                mod.nodes = self.nodes

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

    def build(self, input_shape):
        """ Do the shape inference and initialize weights and bias.

        `build' is a key method in trident, you can use  property `built' to check whether the layer do the build process.
        In build' , we need to put all the logics about  how to comfirm the shape of outputs, weights and bias according to the coming input tensor.

        Args:
            input_shape (tensor):  the shape representation exclude the batch axis.

        """
        pass

    @property
    def trainable_weights(self) -> List[nn.Parameter]:
        """The list of trainable variables (parameters) of the module.
        Parameters of this module and all its submodules are included.

        Notes:
            The list returned may contain duplicate parameters (e.g. output
            layer shares parameters with embeddings). For most usages, it's not
            necessary to ensure uniqueness.

        """
        return [x for x in self.parameters() if x.requires_grad]

    @property
    def non_trainable_weights(self) -> List[nn.Parameter]:
        """
        The list of non-trainable variables (parameters) of the module.Parameters of this module and all its submodules are included.

        Notes:
            The list returned may contain duplicate parameters (e.g. output
            layer shares parameters with embeddings). For most usages, it's not
            necessary to ensure uniqueness.

        """
        return [x for x in self.parameters() if x.requires_grad == False]

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
            return None

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
    def device(self)->str:
        return get_device()

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
        """Shape of input tensor,not including the batch axis."""
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value)->Union[Tensor, Tuple[Tensor]]:
        """ Setting the input_shape, means the layer get shape information and start to do the shape inferrence """

        if isinstance(value, torch.Size):
            value = to_tensor(to_numpy(list(value))).int()
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            value = tuple([to_tensor(to_numpy(list(tensor_shape))).int() if isinstance(tensor_shape, torch.Size) else to_tensor(tensor_shape).int() for tensor_shape in value])

        else:
            value =to_tensor(value).int()

        self.input_spec=TensorSpec(value)
        if self._built == False or  self._input_shape is None:
            self._input_shape = value
            if len(self._input_shape) == 0:
                self.input_filters = int(self._input_shape.data)
            elif len(self._input_shape) == 1:
                self.input_filters = self._input_shape.item()
            else:
                if self.filter_index<0 :
                    self.input_filters = int(self._input_shape[self.filter_index ])
                elif self.filter_index>self.batch_index:
                    self.input_filters = int(self._input_shape[self.filter_index-self.batch_index-1])
                else:
                    raise  NotImplementedError('filter_index>batch_index')

            self.build(self._input_shape)
            self._built = True
            self._signature = None


        elif self._input_shape is not None and to_list(self._input_shape) == to_list(value):
            'input_shape is already assigned, and shape is the same.'
            pass

    @property
    def output_shape(self)->Union[Tensor, Tuple[Tensor]]:
        return self._output_shape

    @output_shape.setter
    def output_shape(self, value):
        if isinstance(value, torch.Size):
            value = to_tensor(to_numpy(list(value))).int()
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            value = tuple([to_tensor(to_numpy(list(tensor_shape))).int() if isinstance(tensor_shape, torch.Size) else to_tensor(tensor_shape).int() for tensor_shape in value])

        else:
            value = to_tensor(value).int()
        self._output_shape = value
        self._signature = None


    @property
    def signature(self):
        if self._signature is None:
            self._signature=Signature(name=self.name)
            if self._input_shape is not None:
                if is_tensor(self._input_shape):
                    self._signature.inputs["input"] = TensorSpec(shape=self._input_shape, name="input")
                elif isinstance(self._input_shape, tuple) and isinstance(self._input_shape[0], int):
                    self._signature.inputs["input"] = TensorSpec(shape=to_tensor(self._input_shape).int(), name="input")
                elif isinstance(self._input_shape, tuple):
                    for i in range(len(self._input_shape)):
                        self._signature.inputs["input_{0}".format(i)] = TensorSpec(shape=self._input_shape[i], name="input_{0}".format(i))
            if self._output_shape is not None:
                if is_tensor(self._output_shape):
                    self._signature.outputs["output"] = TensorSpec(shape=self._output_shape, name="output")
                elif isinstance(self._output_shape, tuple) and isinstance(self._output_shape[0], int):
                    self._signature.outputs["output"] = TensorSpec(shape=to_tensor(self._output_shape).int(), name="output")
                elif isinstance(self._output_shape, tuple):
                    for i in range(len(self._output_shape)):
                        self._signature.outputs["output_{0}".format(i)] = TensorSpec(shape=self._output_shape[i], name="output_{0}".format(i))
        return self._signature

    @signature.setter
    def signature(self, value):
        self._signature=value

    @property
    def input(self):
        return NotImplemented

    @property
    def output(self):
        """
            Retrieves the output tensor(s) of a layer.
            for memory saving issue, we don'tb prefer to keep every input/output
            tensor in every layer.You should set self.keep_output flag to True, and then
            retrive the output tensor when the calll() is executing.
        Returns
            Output tensor or list of output tensors.

        """
        if self.keep_output == False:
            raise ValueError('Layer {0} has not set self.keep_output  to True, cannot access output '.format(self.name))
        return list(self._output_tensor) if isinstance(self._output_tensor, tuple) else self._output_tensor

    def reset_parameters(self):
        pass

    def copy(self):
        """copy the layer

        Returns: The copy of this layer.

        """
        return copy.deepcopy(self)

    def save_onnx(self, file_path=''):
        input_shape = self.input_shape.copy()
        input_shape.insert(0, 1)
        x = torch.randn(*input_shape, requires_grad=False)
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
        is_all_numpy = True
        input = list(input)
        new_input = []
        for inp in input:
            if isinstance(inp, np.ndarray):
                inp = to_tensor(inp)
                new_input.append(inp)
            else:
                new_input.append(inp)
                is_all_numpy = False
        input = new_input
        for hook in self._forward_pre_hooks.values():
            result = hook(self, *input)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                input = result

        if self._built == False:
            inp = unpack_singleton(input)
            if isinstance(inp, (tuple, list)) and all([isinstance(item, numbers.Integral) for item in inp]):
                self.build(inp)
            elif isinstance(inp, (tuple, list)) :
                self.forward(*inp)
            elif is_tensor(inp):
                self.input_shape = tensor_to_shape(inp)
            else:
                print('input shou be tensor or tuple of tensor')
                print(inp)


        if torch._C._get_tracing_state():
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(*input, **kwargs)

            output = unpack_singleton(result)
            if hasattr(self, 'keep_output') and self.keep_output == True:
                self._output_tensor = output
            if isinstance(output, torch.Tensor):# one output
                if self._output_shape is None or not np.array_equal(to_numpy(self._output_shape), to_numpy(int_shape(output))[self.batch_index + 1:]):
                    self._output_shape =tensor_to_shape(output)
            elif isinstance(output, (list, tuple)):
                output_shape = tuple([tensor_to_shape(item) for item in output if not isinstance(item, (list, tuple))])
                # if not isinstance(item, (list,tuple)) lstm
                self._output_shape = unpack_singleton(output_shape)

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
        if is_all_numpy == True and self.training == False:
            if is_tensor(result):
                return to_numpy(result)
            elif isinstance(result, (list, tuple)):
                result = list(result)
            return tuple([to_numpy(res) if is_tensor(res) else res for res in result])
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
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

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
                reset_name(value, self._uid_prefixs)
                value.relative_name = name if not hasattr(value, 'relative_name') or value.relative_name == '' else name + '.' + value.relative_name
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
        super(Sequential, self).__init__()
        self._name = name
        self._built = False
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                module.name = key
                self.add_module(key, module)
        elif len(args) == 1 and isinstance(args[0], (list, nn.ModuleList)):
            for idx, module in enumerate(args[0]):
                self.add_module(str(idx), module)
        else:
            for idx, module in enumerate(args):
                if module._name is not None and len(module._name) > 0:
                    self.add_module(module._name, module)
                else:
                    self.add_module(str(idx), module)
        self.to(self.device)


    def build(self, input_shape):
        """

        Args:
            input_shape (torch.Size, tensor, list(int), tuple(int)): The input_shape information, not including batch axis.

        Returns:

        """
        if self._built == False and len(self._modules) > 0:
            self.__getitem__(0).input_shape = self.input_shape
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
            super(Sequential, self).add_module(name, module)

            dummay_input=random_normal((2,)+tuple(to_list(last_output))).to(self.device)
            out=module(dummay_input)
            self._output_shape =to_tensor(self[-1]._output_shape)
        else:
            super(Sequential, self).add_module(name, module)

        self._signature=None

    def remove_at(self, idx):
        self.__delitem__(idx)
        if len(self._modules) > 0:
            self._output_shape = to_tensor(self[-1]._output_shape)
            self._signature = None

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
        for module in self._modules.values():
            x=enforce_singleton(x)
            x = module(x)
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

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules=None, name=None):
        super(ModuleList, self).__init__()
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

    def __init__(self, *args, name=None):
        super(Combine, self).__init__()
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

    def forward(self, *x):
        outputs = []
        for module in self._modules.values():
            outputs.append(module(*x))
        return tuple(outputs)




def print_network(net, verbose=False):
    num_params = 0
    for i, param in enumerate(net.parameters()):
        num_params += param.numel()
    if verbose:
        logging.info(net)
    logging.info('Total number of parameters: %d\n' % num_params)


def summary(model, input_size, batch_size=-1, device="cuda"):
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
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] =output[0].shape
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            summary[m_key]["flops"] = np.array([0], dtype=np.float64)
            summary[m_key]["macc"] = np.array([0], dtype=np.float64)
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.shape)))
                summary[m_key]["weight"] = list(module.weight.shape)
                summary[m_key]["trainable"] = module.weight.requires_grad
                summary[m_key]["flops"] += (2 * np.prod(np.array(summary[m_key]["weight"]).astype(np.float64)) - 1) * np.prod(
                    np.array(summary[m_key]["output_shape"][2:]).astype(np.float64))
                summary[m_key]["macc"] += np.prod(np.array(summary[m_key]["weight"]).astype(np.float64)) * np.prod(np.array(summary[m_key]["output_shape"][2:]).astype(np.float64))

            if hasattr(module, "bias") and module.bias is not None and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.shape)))
                summary[m_key]["bias"] = list(module.bias.shape)
                summary[m_key]["flops"] += np.prod(np.array(summary[m_key]["bias"]).astype(np.float64)) * np.prod(np.array(summary[m_key]["output_shape"][2:]).astype(np.float64))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, (nn.Sequential, Sequential, nn.ModuleList, ModuleList))
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

    for name, module in model.named_modules():
        module.relative_name = name
    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # prevent pytorch 'ValueError: Expected more than 1 value per channel when training, got input size ....
    model.to(get_device())
    model.eval()
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
    line_new = "{0:^50s} {1:^25s}  {2:^20s} {3:^8s}  {4:^8s}  {5:^25s}".format("Layer (type)", "Output Shape", "Weight ", "Bias", "Param #", "FLOPS #")
    print(line_new)
    print("==============================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    flops = np.array([0], dtype=np.float64)
    macc = 0
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


        line_new=(layer+"  "+class_name).ljust(50,' ')\
                  +(is_keep + str(summary[layer]["output_shape"])).ljust(25,' ')\
                  +str(summary[layer]["weight"] if 'weight' in summary[layer] else '').ljust(20,' ')\
                  +str(summary[layer]["bias"] if 'bias' in summary[layer] else '').ljust(8,' ')\
                  +'{:,}'.format(summary[layer]["nb_params"]).ljust(8,' ')\
                  +'{:,}'.format(summary[layer]["flops"].sum()).ljust(25,' ')

        total_params += summary[layer]["nb_params"]
        flops += float(summary[layer]["flops"])
        macc += float(summary[layer]["macc"].sum())
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)



    # assume 4 bytes/number (float on cuda).
    total_input_size = np.asarray([np.abs(np.prod(to_numpy(shp)) * batch_size * 4. / (1024 ** 2.)) for shp in input_size]).sum()
    total_output_size = np.abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = np.abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("Total MACC: {0:,}".format(int(macc)))
    print("Total FLOPs: {0:.5f} GFLOPs".format(np.round(flops / 10. ** 9, 5)[0]))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary


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





import gc
import subprocess

import numpy as np
import pandas as pd
import torch


class ModelSummary(object):

    def __init__(self, model, mode='full'):
        """
        Generates summaries of model layers and dimensions.
        """
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
        """Run sample input through each layer to get output sizes"""
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
        """Collect Layer Names"""
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
        """Get sizes of all parameters in `model`"""
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
        """Get number of parameters in each layer"""
        param_nums = []
        for mod in self.param_sizes:
            all_params = 0
            for p in mod:
                all_params += np.prod(p)
            param_nums.append(all_params)
        self.param_nums = param_nums
        return

    def make_summary(self):
        """
        Makes a summary listing with:

        Layer Name, Layer Type, Input Size, Output Size, Number of Parameters
        """

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


def try_map_args_and_call(fn, data: OrderedDict, data_feed=None):
    """This function is the core function for mapping callable and argments

    Args:
        fn (callable): the callable, maybe functions or layers
        data (OrderedDict): The key-value pair for available data.
        data_feed (OrderedDict): The relation between callable argments (key) and data (value)

    Returns:
        The result of the callable base on data_feed

    """
    if isinstance(fn, torch.Tensor):
        return fn
    else:
        try:
            arg_map = OrderedDict()
            if isinstance(fn, Layer):
                for arg in fn.signature.inputs.key_list:
                    if arg in data_feed:
                        arg_map[arg] = to_tensor(data[data_feed[arg]]).to(get_device())
                    elif arg in data:
                        arg_map[arg] = to_tensor(data[arg]).to(get_device())
                    else:
                        raise ValueError('arg :{0} cannot mapping correctly!'.format(arg))
                # print('arg_map',arg_map.key_list)
                if get_session_value('amp_available') == True and get_session_value('is_amp_enable') == True:
                    with torch.cuda.amp.autocast():
                        out = fn(*arg_map.value_list)
                else:
                    out = fn(*arg_map.value_list)
                    for item in data.value_list:
                        if hasattr(item, 'cpu'):
                            item.cpu()
                return out
            elif hasattr(fn, 'signature') and callable(fn):
                for arg in fn.signature.inputs.key_list:
                    if arg in data_feed:
                        arg_map[arg] = data[data_feed[arg]].to(get_device())
                    elif arg in data:
                        arg_map[arg] = data[arg].to(get_device())

                    else:

                        raise ValueError('arg :{0} cannot mapping correctly!'.format(arg))
                # print('arg_map', arg_map.key_list)
                if get_session_value('amp_available') == True and get_session_value('is_amp_enable') == True:
                    with torch.cuda.amp.autocast():
                        out = fn(*arg_map.value_list)
                else:
                    out = fn(*arg_map.value_list)
                    for item in data.value_list:
                        if hasattr(item,'cpu'):
                            item.cpu()

                return out
            elif callable(fn):
                args = get_signature(fn).key_list
                for arg in args:
                    if arg in data_feed:
                        arg_map[arg] = data[data_feed[arg]].to(get_device())
                    elif arg in data:
                        arg_map[arg] = data[arg].to(get_device())
                    else:
                        arg_map[arg] = ''
                # print('arg_map', arg_map.key_list)
                if get_session_value('amp_available') == True and get_session_value('is_amp_enable') == True:
                    with torch.cuda.amp.autocast():
                        out = fn(*arg_map.value_list)
                else:
                    out = fn(*arg_map.value_list)
                    for item in data.value_list:
                        if is_tensor(item):
                            item.cpu()
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




def fix_layer(layer:Layer):
    """fix existing out-of-date model compatibility

    Args:
        layer (trident Layer):

    Returns: fixed layer

    """

    if layer._input_shape is not None and isinstance(layer._input_shape,torch.Size):
        layer._input_shape = to_tensor(to_numpy(layer._input_shape)).int()
    if layer._output_shape is not None and isinstance(layer._output_shape,torch.Size):
        layer._output_shape = to_tensor(to_numpy(layer._output_shape)).int()

    for module in layer.modules():
        class_name = module.__class__.__name__
        if not hasattr(module, '_uid_prefixs'):
            module._uid_prefixs = {}
        if not hasattr(module, '_name'):
            module._name = None
            reset_name(module, module._uid_prefixs)

        if not hasattr(module, 'name'):
            module.name=module._name if module._name is not None and len(module._name) > 0 else module.relative_name

        if not hasattr(module, '_built'):
            setattr(module, 'built', True)

        if hasattr(module, 'keepdim'):
            value = getattr(module, 'keepdim')
            delattr(module, 'keepdim')
            setattr(module, 'keepdims', value)

        if not hasattr(module, '_non_persistent_buffers_set'):
            module._non_persistent_buffers_set = set()

        if not hasattr(module, 'input_spec'):
            module.input_spec = None
            if module.input_shape is not None:
                module.input_spec=TensorSpec(shape=module.input_shape)

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

        if 'Conv' in class_name and 'Block' in class_name :
            if not hasattr(module, 'sequence_rank'):
                module.sequence_rank = 'cna'

        if 'Conv' in class_name :
            if not hasattr(module, 'depth_multiplier'):
                if 'Depthwise' in class_name or 'Separable' in class_name :
                    module.depth_multiplier=1
                else:
                    module.depth_multiplier =None
            if not hasattr(module, 'use_spectral'):
                module.use_spectral = False


    if not hasattr(layer,'signature'):
        layer._signature = Signature()
        if layer._input_shape is not None:
            if is_tensor(layer._input_shape):
                layer._signature.inputs["input"] = TensorSpec(shape=layer._input_shape,name="input")
            elif isinstance(layer._input_shape, tuple) and isinstance(layer._input_shape[0], int):
                layer._signature.inputs["input"] = TensorSpec(shape=to_tensor(layer._input_shape).int(), name="input")
            elif isinstance(layer._input_shape, tuple):
                for i in range(len(layer._input_shape)):
                    layer._signature.inputs["input_{0}".format(i)] =  TensorSpec(shape=layer._input_shape[i], name="input_{0}".format(i))
        if layer._output_shape is not None:
            if is_tensor(layer._output_shape):
                layer._signature.outputs["output"] =  TensorSpec(shape=layer._output_shape,name="output")
            elif isinstance(layer._output_shape, tuple) and isinstance(layer._output_shape[0],int):
                layer._signature.outputs["output"] = TensorSpec(shape=to_tensor(layer._output_shape).int(), name="output")
            elif isinstance(layer._output_shape, tuple):
                for i in range(len(layer._output_shape)):
                    layer._signature.outputs["output_{0}".format(i)] =  TensorSpec(shape=layer._output_shape[i], name="output_{0}".format(i))
        layer.signature=layer._signature


    return layer



