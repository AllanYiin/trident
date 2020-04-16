from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import warnings
from collections import OrderedDict
import numpy as np


import cntk
from cntk.layers.blocks import _inject_name,_INFERRED
from cntk.internal import *
from cntk.ops.functions import Function, BlockFunction
from cntk.layers.blocks import _INFERRED
from cntk.default_options import default_override_or, get_default_override
from cntk.layers.blocks import identity, _initializer_for
from cntk.tensor import ArrayMixin,TensorOpsMixin
from cntk.internal import _as_tuple

from ..layers.cntk_activations import *
from ..layers.cntk_normalizations import *
from ..layers.cntk_layers import *
from ..layers.cntk_blocks import *
from data.cntk_datasets import *

version=C.__version__
print('CNTK version:{0}.\n'.format(version))
if float(version)<2.4:
    raise ValueError('Not support CNTK below 2.4' )


dev = C.device.use_default_device()
if dev.type() == 0:
    warnings.warn(
        'CNTK backend warning: GPU is not detected. '
        'CNTK\'s CPU version is not fully optimized,'
        'please run with GPU to get better performance.')



def to_numpy(x) -> np.ndarray:

    """
    Convert whatever to numpy array
    :param x: List, tuple, PyTorch tensor or numpy array
    :return: Numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, C.Parameter):
        return x.value
    elif isinstance(x, ArrayMixin):
        return x.asarray()
    elif isinstance(x, Function):
        return x.eval()
    elif isinstance(x, C.Variable):
        return np.array(x)
    elif isinstance(x, (list, tuple, int, float)):
        return np.array(x)
    else:
        raise ValueError("Unsupported type")

# def to_tensor(x, dtype=None) -> C.T.Tensor:
#     if isinstance(x, torch.Tensor):
#         if dtype is not None:
#             x = x.type(dtype)
#         return x
#
#     if isinstance(x, np.ndarray):
#         x = torch.from_numpy(x)
#         if dtype is not None:
#             x = x.type(dtype)
#         return x
#     if isinstance(x, (list, tuple)):
#         x = np.ndarray(x)
#         x = torch.from_numpy(x)
#         if dtype is not None:
#             x = x.type(dtype)
#         return x
#
#
#
#     raise ValueError("Unsupported input type" + str(type(x)))





def print_summary(model:C.Function, line_length=None, positions=None, print_fn=None):
    """Prints a summary of a model.

    # Arguments
        model: Keras model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
            It defaults to `print` (prints to stdout).
    """
    if print_fn is None:
        print_fn = print  # string to store model

    graph = C.logging.depth_first_search(model.root_function,
                                         lambda x: not isinstance(x, cntk_py.Variable) or not x.is_output, depth=-1)
    names = dict()


    def make_name(n):  # come up with a letter sequence
        if n < 26:
            return chr(n + ord('a'))
        else:
            return make_name(n // 26) + make_name(n % 26)
    def name_it(item):
        if item.name != '':
            return item.name
        if item in names:
            name = names[item]
        else:
            name = make_name(len(names))
            names[item] = name
        return name

    axis_names = dict()
    def name_axis(axis):
        actual_name = axis.name
        if actual_name in axis_names:
            return axis_names[actual_name]
        if axis.name == "UnknownAxes":  # TODO: what is the correct way of testing this?
            name = "?"
        elif axis.name == "defaultBatchAxis":
            name = "b*"
        else:
            name = make_name(len(axis_names) + 12) + "*"
            print("  Axis", actual_name, "==", name)
        axis_names[actual_name] = name
        return name
    def type_spec(var):
        s = "[" + ",".join([name_axis(axis) for axis in var.dynamic_axes]) + "]" if var.dynamic_axes else ''
        s += str(var.shape)
        return s
    def print_item(item):
        name = name_it(item)
        if isinstance(item, cntk_py.Function):
            op_name = item.op_name
            shape = '(' +  ', '.join([name_it(output) + ':' + type_spec(output) for output in item.root_function.outputs]) + ')'
            inputs = '(' +  ', '.join([name_it(input) + ':' + type_spec( input) for input in item.root_function.inputs]) + ')'
            sep = '-> '
        elif isinstance(item, cntk_py.Constant):
            op_name = "Constant"
            shape = type_spec(item)
            inputs = ''
            sep = ''
        elif isinstance(item, cntk_py.Parameter):
            op_name = "Parameter"
            shape = type_spec(item)
            inputs = ''
            sep = ''
        elif isinstance(item, cntk_py.Variable):
            if item.is_parameter:
                op_name = "Parameter"
            elif item.is_placeholder:
                op_name = "Placeholder"
            elif item.is_input:
                op_name = "Input"
            elif item.is_constant:
                op_name = "Constant"
            else:
                op_name = "Variable"
            shape = type_spec(item)
            name = name + " " + item.uid
            sep = ''
            inputs = ''
        print('  {:20} {:30} {} {}{}'.format(op_name, name, inputs, sep, shape))
        pass

    def print_row(fields, positions):
        line = ''
        for i in range(len(fields)):
            if i > 0:
                line = line[:-1] + ' '
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print_fn(line)

    def print_layer_summary(layer):
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = 'multiple'
        name = layer.name
        cls_name = layer.__class__.__name__
        fields = [name + ' (' + cls_name + ')', output_shape, layer.count_params()]
        print_row(fields, positions)

    def print_layer_summary_with_connections(layer):
        """Prints a summary for a single layer.

        # Arguments
            layer: target layer.
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = 'multiple'
        connections = []
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue
            for i in range(len(node.inbound_layers)):
                inbound_layer = node.inbound_layers[i].name
                inbound_node_index = node.node_indices[i]
                inbound_tensor_index = node.tensor_indices[i]
                connections.append(
                    inbound_layer + '[' + str(inbound_node_index) + '][' + str(inbound_tensor_index) + ']')

        name = layer.name
        cls_name = layer.__class__.__name__
        if not connections:
            first_connection = ''
        else:
            first_connection = connections[0]
        fields = [name + ' (' + cls_name + ')', output_shape, layer.count_params(), first_connection]
        print_row(fields, positions)
        if len(connections) > 1:
            for i in range(1, len(connections)):
                fields = ['', '', '', connections[i]]
                print_row(fields, positions)

    #to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Connected to']
    to_display = ['Layer (type)', 'Output Shape', 'Param #']

    print_fn('Model: "{}"'.format(model.name))
    print_fn('_' * line_length)
    print_row(to_display, positions)
    print_fn('=' * line_length)

    relevant_nodes = []
    for v in model._nodes_by_depth.values():
        relevant_nodes += v
    sequential_like=True
    for i in range(len(graph)):
        item=graph[i]
        print_item(item)
        if sequential_like:
            print_layer_summary(item)
        else:
            print_layer_summary_with_connections(item)
        if i == len(graph) - 1:
            print_fn('=' * line_length)
        else:
            print_fn('_' * line_length)

    trainable_count=0
    non_trainable_count=0
    if any(any(dim == InferredDimension for dim in p.shape) for p in model.parameters):
        total_parameters = 'so far unspecified number of'
    else:
        total_parameters = sum([reduce(mul, p.shape + (1,)) for p in model.parameters ])
        # the +(1,) is needed so that this works for empty shapes (scalars)
    print("Training {} parameters in {} parameter tensors.".format(total_parameters, len(model.parameters)))

    print_fn('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print_fn('Trainable params: {:,}'.format(trainable_count))
    print_fn('Non-trainable params: {:,}'.format(non_trainable_count))
    print_fn('_' * line_length)





def summary(model,input_shape=None,device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(module.arguments[0].shape)
            summary[m_key]["input_shape"].insert(0,-1)
            output=module.outputs
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.shape)
                summary[m_key]["output_shape"].insert(0,-1)

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, C.layers.Sequential)
            # and not isinstance(module, C.Variable)
            and not (module == C.Constant)
        ):
            hooks.append(module.register_forward_hook(hook))


    graph = C.logging.depth_first_search(model.root_function,
                                         lambda x: not isinstance(x, cntk_py.Variable) or not x.is_output, depth=-1)
    names = dict()

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    filter_shape = _as_tuple(filter_shape)
    num_filters = _as_tuple(num_filters or ())
    filter_rank = len(filter_shape)
    strides = _pad_to_shape(filter_shape, strides, 'strides')
    sharing = _pad_to_shape(filter_shape, sharing, 'sharing')
    pad = _pad_to_shape(filter_shape, pad, 'pad')
    dilation = _pad_to_shape(filter_shape, dilation, 'dilation')
    # multiple inputs to the network
    if input_shape is None:
        input_shape=model.arguments[0].shape
    # batch_size of 2 for batchnorm
    x =np.random.standard_normal(input_shape)
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

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    trainable_params = 0
    non_trainable_params = 0
    if any(any(dim == _InferredDimension for dim in p.shape) for p in model.parameters):
        total_parameters = 'so far unspecified number of'
    else:
        total_parameters = sum([reduce(mul, p.shape + (1,)) for p in
                                model.parameters])  # the +(1,) is needed so that this works for empty shapes (scalars)
    print("Training {} parameters in {} parameter tensors.".format(total_parameters, len(model.parameters)))



    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")

    # return summary
