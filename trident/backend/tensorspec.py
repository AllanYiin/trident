import copy
import inspect
import numbers
import sys
from collections import Counter
from enum import Enum
from inspect import signature
from trident.backend.common import to_list, OrderedDict, Signature, split_path, unpack_singleton, get_session, get_backend, TensorShape, is_instance
from trident.backend import dtype
from typing import Optional, Union, overload
import numpy as np

__all__ = ['TensorSpec', 'ObjectType', 'assert_input_compatibility', 'assert_spec_compatibility', 'get_python_function_arguments', 'get_signature', 'ExpectDataType',
           'object_type_inference', 'distict_color_count']

if get_backend() == 'pytorch':
    import torch
    Tensor = torch.Tensor
    from trident.backend.pytorch_ops import *
elif get_backend() == 'tensorflow':
    import tensorflow as tf
    Tensor=tf.Tensor
    from trident.backend.tensorflow_ops import *



_primes = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
           173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251]


class ObjectType(Enum):
    array_data = 'array_data'
    gray = 'gray_image'
    rgb = 'rgb_image'
    rgba = 'rgba_image'
    image3d='image3d'
    image_path='image_path'
    label_mask = 'label_mask'
    color_mask = 'color_mask'
    binary_mask = 'binary_mask'
    alpha_mask = 'alpha_mask'
    multi_channel = 'multi_channel'
    absolute_bbox = 'absolute_bbox'
    relative_bbox = 'relative_bbox'
    landmarks = 'landmarks'
    random_noise = 'random_noise'
    classification_label = 'classification_label'
    corpus = 'corpus'
    sequence_label = 'sequence_label'
    sequence_mask = 'sequence_mask'
    embedding = 'embedding'


def distict_color_count(img):
    """Count for distinct colors in input image

    Returns:
        object:

    Examples:
        >>> from trident import *
        >>> img=read_image('../../trident_logo.png')[:,:,:3]
        >>> color_counter=distict_color_count(img)
        >>> len(color_counter)
        1502
        >>> img=read_image('../../trident_logo.png')[:,:,:1]
        >>> len(distict_color_count(img))
        125

    """
    return Counter([tuple(colors) for i in img for colors in i])


def pixel_numbers(data):
    return list(set(data.reshape(-1).tolist()))


def object_type_inference(data):
    if isinstance(data, np.ndarray):
        if data.ndim == 2 and data.shape[-1] == 2:
            return ObjectType.landmarks
        elif data.ndim == 2 and data.shape[-1] in (4, 5) and 0 <= data.max().round(0) <= 255 and 0 <= data.min().round(0) <= 255:
            return ObjectType.absolute_bbox
        elif data.ndim == 2 and data.shape[-1] in (4, 5) and 0 <= data.max().round(0) <= 1 and 0 <= data.min().round(0) <= 1:
            return ObjectType.relative_bbox
        elif data.ndim == 2 and len(distict_color_count(np.expand_dims(data, -1))) == 2:
            return ObjectType.binary_mask
        elif data.ndim == 2 and (data.max() - data.min() + 1) == len(distict_color_count(np.expand_dims(data, -1))):
            return ObjectType.label_mask
        elif data.ndim == 2 and 0 <= data.max().round(0) <= 255 and 0 <= data.min().round(0) <= 255:
            return ObjectType.gray
        elif data.ndim == 3 and (data.shape[-1] == 1 or data.shape[0] == 1) and len(distict_color_count(data)) == 2:
            return ObjectType.binary_mask
        elif data.ndim == 3 and (data.shape[-1] == 1 or data.shape[0] == 1) and (data.max() - data.min() + 1) == len(distict_color_count(data)):
            return ObjectType.label_mask
        elif data.ndim == 3 and (data.shape[-1] == 1 or data.shape[0] == 1) and 0 <= data.max().round(0) <= 255 and 0 <= data.min().round(0) <= 255:
            return ObjectType.gray
        elif data.ndim == 3 and (data.shape[-1] == 3 or data.shape[0] == 3) and 0 <= data.max().round(0) <= 255 and 0 <= data.min().round(0) <= 255:
            if np.array_equal(data[:, :, 0], data[:, :, 1]) and np.array_equal(data[:, :, 2], data[:, :, 1]):
                return ObjectType.gray
            elif np.array_equal(data[0, :, :], data[0, :, :]) and np.array_equal(data[2, :, :], data[2, :, :]):
                return ObjectType.gray
            elif not np.array_equal(data, data.round()):
                return ObjectType.rgb
            elif len(list(set(pixel_numbers(data)) & set(_primes[3:]))) > 3:
                return ObjectType.rgb
            elif len(distict_color_count(data)) < 100:
                return ObjectType.color_mask
            else:
                return ObjectType.rgb
        elif data.ndim == 3 and data.shape[-1] == 4 and 0 <= data.max().round(0) <= 255 and 0 <= data.min().round(0) <= 255:
            return ObjectType.rgba
        elif data.ndim == 3 and data.dtype == np.int64 and 0 <= data.max().round(0) <= 1 and 0 <= data.min().round(0) <= 1:
            return ObjectType.binary_mask
        elif data.ndim == 3 and data.dtype in [np.float32, np.float16] and 0 <= data.max() <= 1 and 0 <= data.min().round(0) <= 1:
            return ObjectType.alpha_mask
        elif data.ndim <= 1 and data.dtype == np.int64:
            return ObjectType.classification_label
        elif data.ndim == 2 and data.dtype == np.int64:
            return ObjectType.color_mask
        else:
            sys.stderr.write('Object type cannot be inferred: shape:{0} dtype:{1} min:{2} max:{3} .'.format(data.shape, data.dtype, data.min(), data.max()) + '\n')
            return ObjectType.array_data
    elif isinstance(data, list):
        data = unpack_singleton(data)
        if isinstance(data, str):
            return ObjectType.corpus
        elif len(data) > 0 and isinstance(data[0], str):
            return ObjectType.corpus
    elif isinstance(data, str):
        return ObjectType.corpus


ExpectDataType = ObjectType


class TensorSpec(object):
    """Specifies the rank, dtype and shape of every input to a layer.
    Layers can expose (if appropriate) an `input_spec` attribute:
    an instance of `InputSpec`, or a nested structure of `InputSpec` instances
    (one per input tensor). These objects enable the layer to run input
    compatibility checks for input structure, input rank, input shape, and
    input dtype.
    A None entry in a shape is compatible with any dimension,
    a None shape is compatible with any shape.
    Arguments:
        dtype: Expected DataType of the input.
        shape: Shape tuple, expected shape of the input
            (may include None for unchecked axes).
        ndim: Integer, expected rank of the input.
        max_ndim: Integer, maximum rank of the input.
        min_ndim: Integer, minimum rank of the input.
        axes: Dictionary mapping integer axes to
            a specific dimension value.

  Examples:
      >>> t=np.random.standard_normal((2,3,4))
      >>> spec=TensorSpec(shape=t.shape)
      >>> print(spec.__class__.__name__)
      TensorSpec
      >>> print(spec)
      TensorSpec(shape=(None, 2, 3, 4), ndim=3)
      >>> t1=cast(arange(10).reshape(1,2,5),'float16')
      >>> print(t1.dtype)
      float16
      >>> TensorSpec.tensor_to_spec(t1)
      TensorSpec(dtype=torch.float32, shape=(None, 2, 5), ndim=3)
    """

    def __init__(self,
                 shape: Union[None, TensorShape] = None,
                 ndim: Union[None, int] = None,
                 max_ndim: Union[None, int] = None,
                 min_ndim: Union[None, int] = None,
                 axes=None,
                 dtype=None,
                 object_type: Optional[ObjectType] = None,
                 optional=False,
                 default=None,
                 name=None,**kwargs):
        self._dtype = dtype if dtype is not None else None
        self._shape_tuple = None
        self.object_type = object_type
        self.optional = optional
        self.default = default
        self._name = name
        if shape is not None:
            if isinstance(shape, TensorShape):
                self.ndim = shape.ndims
                self._shape_tuple = tuple(shape.dims)
                self.shape = shape
            elif isinstance(shape, (list, tuple)) and all([isinstance(item, numbers.Number) for item in shape]):
                self.ndim = len(shape)
                self._shape_tuple = (None,) + tuple(shape)
                self.shape = TensorShape(shape)
            elif not is_tensor(shape) and isinstance(shape, numbers.Number):
                self.ndim = 0
                self.shape = TensorShape([None, shape])
                self._shape_tuple = (None, shape)
            elif is_tensor(shape) and 'int' in str(shape.dtype):
                self.ndim = len(shape)

                shape = to_list(to_numpy(shape))
                self._shape_tuple = (None,) + tuple(shape)
                self.shape = TensorShape(self._shape_tuple)
            else:
                print(shape)
                self.ndim = len(shape)
                shape = to_list(to_numpy(shape))
                self._shape_tuple = (None,) + tuple(shape)
                self.shape = TensorShape(self._shape_tuple)
        else:
            self.ndim = ndim
            self.shape = None

        self.max_ndim = max_ndim
        self.min_ndim = min_ndim
        try:
            axes = axes or {}
            self.axes = {int(k): axes[k] for k in axes}
        except (ValueError, TypeError):
            raise TypeError('The keys in axes must be integers.')

        if self.axes and (self.ndim is not None or self.max_ndim is not None):
            max_dim = (self.ndim if self.ndim else self.max_ndim) - 1
            max_axis = max(self.axes)
            if max_axis > max_dim:
                raise ValueError('Axis {} is greater than the maximum allowed value: {}'
                                 .format(max_axis, max_dim))

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in ['shape']:
            if value is not None and value != TensorShape([None]) and value != TensorShape([0]):
                self.ndim = self.shape.ndims
                self._shape_tuple = tuple(self.shape.dims)

    def copy(self):
        return copy.deepcopy(self)

    @classmethod
    def tensor_to_spec(cls, t, object_type: ObjectType = None, need_exclude_batch_axis=True, is_singleton=False, optional=False, name=None):
        if isinstance(t,str):
            return cls(shape=TensorShape([None]), dtype=str, object_type=object_type,
                       optional=optional,
                       name=name)
        else:
            t = to_tensor(t)
            return cls(shape=tensor_to_shape(t, need_exclude_batch_axis=need_exclude_batch_axis, is_singleton=is_singleton), dtype=t.dtype, object_type=object_type, optional=optional,
                   name=name)

    @property
    def dtype(self):
        """Returns the `dtype` of elements in the tensor."""
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        return value

    @property
    def name(self):
        """Returns the (optionally provided) name of the described tensor."""
        return self._name

    @property
    def is_spatial(self):
        if self.object_type is not None:
            if 'mask' in self.object_type.value or 'bbox' in self.object_type.value or 'rgb' in self.object_type.value or self.object_type == ObjectType.gray or self.object_type\
                    == ObjectType.landmarks:
                return True
        return False

    def is_compatible_with(self, inputs):  # pylint:disable=useless-super-delegation
        """Returns True if spec_or_tensor is compatible with this TensorSpec.
        Two tensors are considered compatible if they have the same dtype
        and their shapes are compatible (see `TensorShape.is_compatible_with`).
        Args:
          inputs: A TensorSpec or a Tensor
        Returns:
          True if spec_or_tensor is compatible with self.
        """
        if self.shape is None:
            return False
        if len(inputs) != len(self):
            raise ValueError('Expects ' +
                             str(len(self)) + ' inputs, '
                                              'but it received ' + str(len(inputs)) +
                             ' input tensors. Inputs received: ' + str(inputs))
        for input_index, (x, spec) in enumerate(zip(inputs, self)):
            if spec is None:
                continue

            if (spec.ndim is not None or
                    spec.min_ndim is not None or
                    spec.max_ndim is not None):
                if x.shape.ndims is None:
                    raise ValueError('Input ' + ' is incompatible : '
                                                'its rank is undefined, but the layer requires a '
                                                'defined rank.')

            # Check ndim.
            if spec.ndim is not None:
                ndim = x.shape.ndims
                if ndim != spec.ndim:
                    raise ValueError('Input ' + str(input_index) + ' is incompatible with the layer: '
                                                                   'expected ndim=' + str(spec.ndim) + ', found ndim=' +
                                     str(ndim) + '. Full shape received: ' +
                                     str(x.shape.as_list()))
            if spec.max_ndim is not None:
                ndim = x.shape.ndims
                if ndim is not None and ndim > spec.max_ndim:
                    raise ValueError('Input ' + str(input_index) + ' is incompatible with the layer: '
                                                                   'expected max_ndim=' + str(spec.max_ndim) +
                                     ', found ndim=' + str(ndim))
            if spec.min_ndim is not None:
                ndim = x.shape.ndims
                if ndim is not None and ndim < spec.min_ndim:
                    raise ValueError('Input ' + str(input_index) + ' is incompatible with the layer: '
                                                                   ': expected min_ndim=' + str(spec.min_ndim) +
                                     ', found ndim=' + str(ndim) +
                                     '. Full shape received: ' +
                                     str(x.shape.as_list()))
            # Check dtype.
            if spec.dtype is not None:
                if x.dtype != spec.dtype:
                    raise ValueError('Input ' + str(input_index) + ' is incompatible with the layer: '
                                                                   'expected dtype=' + str(spec.dtype) +
                                     ', found dtype=' + str(x.dtype))
            # Check specific shape axes.
            if spec.axes:
                shape = x.shape.as_list()
                if shape is not None:
                    for axis, value in spec.axes.items():
                        if hasattr(value, 'value'):
                            value = value.value
                        if value is not None and shape[int(axis)] not in {value, None}:
                            raise ValueError(
                                'Input ' + str(input_index) + ' is'
                                                              ' incompatible with the layer: expected axis ' + str(axis) +
                                ' of input shape to have value ' + str(value) +
                                ' but received input with shape ' + str(shape))
            # Check shape.
            if spec.shape is not None:
                shape = x.shape.as_list()
                if shape is not None:
                    for spec_dim, dim in zip(spec.shape, shape):
                        if spec_dim is not None and dim is not None:
                            if spec_dim != dim:
                                raise ValueError('Input ' + str(input_index) +
                                                 ' is incompatible  ' +
                                                 ': expected shape=' + str(spec.shape) +
                                                 ', found shape=' + str(shape))

    def __hash__(self):
        return hash((self._shape_tuple, self.dtype))

    def __eq__(self, other):
        # pylint: disable=protected-access
        return (type(self) is type(other) and
                self._shape_tuple == other._shape_tuple
                and self._dtype == other._dtype
                and self.name == other.name
                and self.object_type == other.object_type)

    def __repr__(self):
        spec = [('dtype=' + str(self.dtype)) if self.dtype else '',
                ('shape=' + str(self._shape_tuple)) if self._shape_tuple else '',
                ('ndim=' + str(self.ndim)) if self.ndim else '',
                ('max_ndim=' + str(self.max_ndim)) if self.max_ndim else '',
                ('min_ndim=' + str(self.min_ndim)) if self.min_ndim else '',
                ('axes=' + str(self.axes)) if self.axes else '',
                ('object_type=' + str(self.object_type)) if self.object_type else '',
                ('name=' + str(self._name)) if self._name else '',
                ]
        return 'TensorSpec(%s)' % ', '.join(x for x in spec if x)


def assert_input_compatibility(input_spec: TensorSpec, inputs):
    """Checks compatibility between the layer and provided inputs.
    This checks that the tensor(s) `inputs` verify the input assumptions
    of a layer (if any). If not, a clear and actional exception gets raised.
    Arguments:
        input_spec: An InputSpec instance, list of InputSpec instances, a nested
            structure of InputSpec instances, or None.
        inputs: Input tensor, list of input tensors, or a nested structure of
            input tensors.

    Raises:
        ValueError: in case of mismatch between
            the provided inputs and the expectations of the layer.
    """
    if not input_spec:
        return
    input_spec.shape.to('cpu')
    inputs.to('cpu')
    if len(inputs) != len(input_spec):
        raise ValueError('Tensor ' + ' expects ' +
                         str(len(input_spec)) + ' inputs, '
                                                'but it received ' + str(len(inputs)) +
                         ' input tensors. Inputs received: ' + str(inputs))
    for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
        if spec is None:
            continue

        if (spec.ndim is not None or
                spec.min_ndim is not None or
                spec.max_ndim is not None):
            if x.shape.ndims is None:
                raise ValueError('Input ' + str(input_index) + ' of tensor ' + ' is incompatible with the layer: '
                                                                               'its rank is undefined, but the layer requires a '
                                                                               'defined rank.')

        # Check ndim.
        if spec.ndim is not None:
            ndim = x.shape.ndims
            if ndim != spec.ndim:
                raise ValueError('Input ' + str(input_index) + ' of tensor ' + ' is incompatible with the layer: '
                                                                               'expected ndim=' + str(spec.ndim) + ', found ndim=' +
                                 str(ndim) + '. Full shape received: ' +
                                 str(x.shape.as_list()))
        if spec.max_ndim is not None:
            ndim = x.shape.ndims
            if ndim is not None and ndim > spec.max_ndim:
                raise ValueError('Input ' + str(input_index) + ' of tensor ' + ' is incompatible with the layer: '
                                                                               'expected max_ndim=' + str(spec.max_ndim) +
                                 ', found ndim=' + str(ndim))
        if spec.min_ndim is not None:
            ndim = x.shape.ndims
            if ndim is not None and ndim < spec.min_ndim:
                raise ValueError('Input ' + str(input_index) + ' of tensor ' + ' is incompatible with the layer: '
                                                                               ': expected min_ndim=' + str(spec.min_ndim) +
                                 ', found ndim=' + str(ndim) +
                                 '. Full shape received: ' +
                                 str(x.shape.as_list()))
        # Check dtype.
        if spec.dtype is not None:
            if x.dtype != spec.dtype:
                raise ValueError('Input ' + str(input_index) + ' of tensor ' + ' is incompatible with the layer: '
                                                                               'expected dtype=' + str(spec.dtype) +
                                 ', found dtype=' + str(x.dtype))
        # Check specific shape axes.
        if spec.axes:
            shape = x.shape.as_list()
            if shape is not None:
                for axis, value in spec.axes.items():
                    if hasattr(value, 'value'):
                        value = value.value
                    if value is not None and shape[int(axis)] not in {value, None}:
                        raise ValueError(
                            'Input ' + str(input_index) + ' of tensor ' + ' is'
                                                                          ' incompatible with the layer: expected axis ' + str(axis) +
                            ' of input shape to have value ' + str(value) +
                            ' but received input with shape ' + str(shape))
        # Check shape.
        if spec.shape is not None:
            shape = x.shape.as_list()
            if shape is not None:
                for spec_dim, dim in zip(spec.shape, shape):
                    if spec_dim is not None and dim is not None:
                        if spec_dim != dim:
                            raise ValueError('Input ' + str(input_index) +
                                             ' is incompatible with tensor ' +
                                             ': expected shape=' + str(spec.shape) +
                                             ', found shape=' + str(shape))


def assert_spec_compatibility(input_spec: TensorSpec, other_spec: TensorSpec):
    """Checks compatibility between the layer and provided inputs.
    This checks that the tensor(s) `inputs` verify the input assumptions
    of a layer (if any). If not, a clear and actional exception gets raised.
    Arguments:
        input_spec: An InputSpec instance, list of InputSpec instances, a nested
            structure of InputSpec instances, or None.
        other_spec: Another InputSpec

    Raises:
        ValueError: in case of mismatch between
            the provided inputs and the expectations of the layer.
    """
    if not input_spec:
        return False
    if isinstance(input_spec, (tuple, list)) and all([isinstance(item, numbers.Integral) for item in input_spec]):
        input_spec = TensorSpec(shape=to_tensor(input_spec))

    if isinstance(other_spec, (tuple, list)) and all([isinstance(item, numbers.Integral) for item in other_spec]):
        other_spec = TensorSpec(shape=to_tensor(other_spec))

    if (input_spec.ndim is not None or
            input_spec.min_ndim is not None or
            input_spec.max_ndim is not None):
        if other_spec.ndim is None:
            print('Other_spec ' + ' is incompatible with input_spec: '
                                  'its rank is undefined, but input_spec requires a '
                                  'defined rank.')
            return False

    # Check ndim.
    if input_spec.ndim is not None:
        ndim = other_spec.ndim
        if ndim != input_spec.ndim:
            print('Other_spec is incompatible with the input_spec: expected ndim=' + str(input_spec.ndim) + ', found ndim=' +
                  str(ndim) + '. Full shape received: ' +
                  str(other_spec._shape_tuple))
            return False
    if input_spec.max_ndim is not None:
        ndim = other_spec.ndim
        if ndim is not None and ndim > input_spec.max_ndim:
            print('Other_spec is incompatible with the input_spec: expected max_ndim=' + str(input_spec.max_ndim) +
                  ', found ndim=' + str(ndim))
            return False
    if input_spec.min_ndim is not None:
        ndim = other_spec.ndim
        if ndim is not None and ndim < input_spec.min_ndim:
            print('Other_spec is incompatible with the input_spec: expected min_ndim=' + str(input_spec.min_ndim) +
                  ', found ndim=' + str(ndim) +
                  '. Full shape received: ' +
                  str(other_spec._shape_tuple))
            return False
    # Check dtype.
    if input_spec.dtype is not None:
        if other_spec.dtype != input_spec.dtype:
            print('Other_spec is incompatible with the input_spec: expected dtype=' + str(input_spec.dtype) +
                  ', found dtype=' + str(other_spec.dtype))
            return False
    # Check specific shape axes.
    if input_spec.axes:
        shape = other_spec._shape_tuple
        if shape is not None:
            for axis, value in input_spec.axes.items():
                if hasattr(value, 'value'):
                    value = value.value
                if value is not None and shape[int(axis)] not in {value, None}:
                    print(
                        'Other_spec is  incompatible with input_spec: expected axis ' + str(axis) +
                        ' of input shape to have value ' + str(value) +
                        ' but received input with shape ' + str(shape))
                    return False
    # Check shape.
    if input_spec.shape is not None:
        shape = other_spec._shape_tuple
        is_compatible = TensorShape(input_spec.shape).is_compatible_with(TensorShape(other_spec._shape_tuple))
        if is_compatible:
            return is_compatible
        if shape is not None:
            for spec_dim, dim in zip(other_spec._shape_tuple, input_spec._shape_tuple):
                if spec_dim is not None and dim is not None:
                    if spec_dim != dim:
                        print('Other_spec is incompatible with input_spec: expected shape=' + str(input_spec._shape_tuple) +
                              ', found shape=' + str(shape))
                        return False
    return True


def get_python_function_arguments(f):
    """
    Helper to get the parameter names and annotations of a Python function.
    Examples:
    """
    # Note that we only return non-optional arguments (we assume that any optional args are not specified).
    # This allows to, e.g., accept max(a, b, *more, name='') as a binary function
    param_specs = inspect.getfullargspec(f)
    annotations = param_specs.annotations
    arg_names = param_specs.args
    defaults = param_specs.defaults  # "if this tuple has n elements, they correspond to the last n elements listed
    # in args"
    if defaults:
        arg_names = arg_names[:-len(defaults)]
    return (arg_names, annotations)


def get_signature(fn, name=None):
    """

    Args:
        name ():
        fn ():


    Returns:

    Examples:
        >>> get_signature(unpack_singleton)
        split_path( path:<class 'str'>) -> folder, filename, ext


    """
    # if is_instance(fn,'Sequential') and len(fn)>0:
    #     sig1=get_signature(fn[0])
    #     sig2 =get_signature(fn[-1])
    #     sig1.outputs=sig2.outputs
    #     return sig1


    base_fn = fn
    signature = base_fn._signature if hasattr(fn, '_signature') else  None
    if signature is None:
        signature=Signature()
    if hasattr(fn, 'forward'):
        base_fn = fn.forward
    sig = inspect.signature(base_fn)
    paras = list(sig.parameters.items())

    returns = sig.return_annotation
    if sig.return_annotation is not inspect._empty and returns is not None:
        if isinstance(returns, str):
            signature.outputs[returns] = TensorSpec(TensorShape([None]), optional=False, name=returns)
        elif returns is Tensor:
            signature.outputs['output'] = TensorSpec(TensorShape([None]), optional=False, name='output')
        else:
            if returns is not None:
                for i in range(len(returns)):
                    signature.outputs['output_{0}'.format(i)] = TensorSpec(TensorShape([None]), optional=False, name='output_{0}'.format(i))
    else:
        signature.outputs['output'] = TensorSpec(TensorShape([None]), optional=False, name='output')

    for k, v in paras:
        if k not in ['kwargs', 'self', 'args']:
            annotation = v.annotation
            _default = v.default if v.default != inspect.Parameter.empty else None
            _optional = False if v.default == inspect.Parameter.empty else True
            _dtype = None if _default is None else type(_default) if _default is not None  else annotation if not is_instance(annotation, str) else None
            _ndim = 0 if _dtype in [int, float, bool, str, numbers.Number, numbers.Integral] else None
            _shape = TensorShape([0]) if _dtype in [int, float, bool, str, numbers.Number, numbers.Integral] else None
            if annotation is Tensor:
                _dtype = dtype.float32
                _shape = TensorShape([None])
                _ndim = None
            signature.inputs[k] = TensorSpec(shape=_shape, ndim=_ndim, dtype=_dtype, optional=_optional, default=_default, name=k)

    if name is not None:
        signature.name = name
    else:
        signature.name = name if name is not None else fn.__class__.__name__ if is_instance(fn, 'Layer') else fn.__class__.__qualname__
    # if hasattr(base_fn,'__code__'):
    #     signature.__code__ = base_fn.__code__
    return signature


def update_signature(fn: callable, args: list):
    sig = None
    if hasattr(fn, 'signature') and fn.signature is not None:
        sig = fn.signature
    else:
        sig = get_signature(fn)

    new_sig = Signature(name=sig.name)
    for i in range(len(args)):
        new_sig.inputs[args[i]] = sig.inputs.value_list[i]
    new_sig.outputs = sig.outputs
    fn.signature = new_sig
    print(fn.signature)





