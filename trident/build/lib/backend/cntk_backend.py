from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .cntk_ops import *
from layers.cntk_activations import *
from layers.cntk_normalizations import *
from layers.cntk_blocks import *
from data.cntk_datasets import *

version=C.__version__
print('CNTK version:{0}.\n'.format(version))
if float(version)<2.4:
    raise ValueError('Not support CNTK below 2.4' )






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

    graph = C.logging.depth_first_search(model.root_function,lambda x: not isinstance(x, cntk_py.Variable) or not x.is_output,depth=-1)
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

