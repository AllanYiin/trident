from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import itertools
import threading
import weakref

import numpy as np
import six

from trident import context
from trident.backend import iteration_tools
from trident.backend.common import *
from trident.backend.tensorspec import TensorSpec

__all__ = ['FunctionSpec', 'Function']

tctx = context._context()
_backend = tctx.get_backend()
working_directory = tctx.working_directory
if _backend == 'pytorch':

    from trident.backend.pytorch_backend import *
    from trident.backend.pytorch_ops import *

elif _backend == 'tensorflow':
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *


def _deterministic_dict_values(dictionary):
    """Deterministic dictionary values.

    Args:
        dictionary: A dictionary.

    Returns:
        A tuple of values from the dictionary, sorted by key."""
    return tuple(dictionary[key] for key in sorted(dictionary))


def _convert_numpy_inputs(inputs):
    """Convert numpy array inputs to tensors."""
    # We assume that any CompositeTensors have already converted their components
    # from numpy arrays to Tensors, so we don't need to expand composites here for
    # the numpy array conversion. Instead, we do so because the flattened inputs
    # are eventually passed to ConcreteFunction()._call_flat, which requires
    # expanded composites.
    flat_inputs = iteration_tools.flatten(inputs)

    # Check for NumPy arrays in arguments and convert them to Tensors.
    # TODO(nareshmodi): Skip ndarray conversion to tensor altogether, perhaps
    # finding a way to store them directly in the cache key (currently not
    # possible since ndarrays are not hashable).
    need_packing = False
    filtered_flat_inputs = []
    for index, value in enumerate(flat_inputs):
        if isinstance(value, (Tensor, Parameter)):
            filtered_flat_inputs.append(value)
        elif hasattr(value, "__array__") and not (hasattr(value, "_should_act_as_resource_variable") or
                                                  isinstance(value, (np.str_, type, CompositeTensor))):
            # This case is equivalent to _is_ndarray(value) == True
            a = to_numpy(value)
            if not isinstance(a, np.ndarray):
                raise TypeError("The output of __array__ must be an np.ndarray "
                                "(got {} from {}).".format(type(a), type(value)))
            flat_inputs[index] = to_tensor(a, requires_grad=False)  # constant_op.constant(a)
            filtered_flat_inputs.append(flat_inputs[index])
            need_packing = True
    if need_packing:
        return (iteration_tools.pack_sequence_as(
            structure=inputs, flat_sequence=flat_inputs,
            expand_composites=True), flat_inputs, filtered_flat_inputs)
    else:
        return inputs, flat_inputs, filtered_flat_inputs


def _convert_inputs_to_signature(inputs, input_signature, flat_input_signature):
    """Convert inputs to pass into a function with an explicit signature."""

    def format_error_message(inputs, input_signature):
        """Formats an error message.

        Args:
            inputs: List of inputs
            input_signature: List of input signatures

        Returns:
            Formatted error message"""
        return ("  inputs: (\n" + "    " + ",\n    ".join(str(i) for i in inputs) +
                ")\n" + "  input_signature: (\n" + "    " +
                ",\n    ".join(str(i) for i in input_signature) + ")")

    try:
        flatten_inputs = iteration_tools.flatten_up_to(
            input_signature,
            inputs[:len(input_signature)],
            expand_composites=True,
            check_types=False)  # lists are convert to tuples for `tf.data`.
    except ValueError:
        raise ValueError("Structure of Python function inputs does not match "
                         "input_signature:\n%s" %
                         format_error_message(inputs, input_signature))

    need_packing = False
    for index, (value, spec) in enumerate(zip(flatten_inputs,
                                              flat_input_signature)):
        if (isinstance(spec, TensorSpec) and
                not is_tensor(value)):
            try:
                flatten_inputs[index] = to_tensor(value, dtype=spec.dtype)
                need_packing = True
            except ValueError:
                raise ValueError("When input_signature is provided, all inputs to "
                                 "the Python function must be convertible to "
                                 "tensors:\n%s" %
                                 format_error_message(inputs, input_signature))

    if any(not spec.is_compatible_with(other) for spec, other in zip(
            flat_input_signature,
            flatten_inputs)):
        raise ValueError("Python inputs incompatible with input_signature:\n%s" %
                         format_error_message(inputs, input_signature))

    if need_packing:
        inputs = iteration_tools.pack_sequence_as(
            structure=input_signature,
            flat_sequence=flatten_inputs,
            expand_composites=True)

    flat_inputs = iteration_tools.flatten(inputs)

    return (inputs, flat_inputs, [
        t for t in flat_inputs
        if isinstance(t, (Tensor, Parameter))
    ])


class FunctionSpec(object):
    """Specification of how to bind arguments to a function."""

    @staticmethod
    def from_function_and_signature(python_function,
                                    input_signature,
                                    is_pure=False,
                                    experimental_follow_type_hints=False,
                                    experimental_compile=None):
        """Create a FunctionSpec instance given a python function and signature.

    Args:
      python_function: a function to inspect
      input_signature: a signature of the function (None, if variable)
      is_pure: if True all input arguments (including variables and constants)
      will be converted to tensors and no variable changes allowed.
      experimental_follow_type_hints: see `tf.function`
      experimental_compile: see `tf.function`

    Returns:
      instance of FunctionSpec
    """
        fullargspec = inspect.getfullargspec(python_function)
        # Treat a wrapped partial function as a special case. For all arguments that
        # were overridden with keywords in the partial:
        #   - remove the corresponding arguments,
        #   - remove the corresponding keywords.
        # _, unwrapped = unwrap(python_function)
        # TODO(b/131153379): Consider Python3's fullargspec.kwonlyargs and
        # fullargspec.kwonlydefaults.
        if isinstance(python_function, functools.partial):
            # Also consider the Python3 case with kwonlydefaults.
            if fullargspec.defaults or fullargspec.kwonlydefaults:
                new_defaults = fullargspec.defaults
                new_args = fullargspec.args
                if fullargspec.defaults:
                    # To be able to canonicalize the function properly, we want to ignore
                    # default values that are overridden via a partial kwarg. For example:
                    #
                    #   def func(a, b, c, d=5, e=7):
                    #     return a, b, c, d, e
                    #   p_func = functools.partial(tf.function(func, 10, e=9))
                    #
                    # Here we want to drop from the defaults the parameter `e`. If we
                    # forwarded the call to the partial function with a default for `e`
                    # we would get an error for passing two values for one parameter.
                    #
                    # Note that this has a limitation: we can only override parameters at
                    # the end of the parameter list.
                    #
                    # In this case we want to end up with 3 arguments (b, c, d) and 1
                    # default value (5). We do this by constructing a mask where 0 stands
                    # for a value that was overridden by a partial kwarg. The seemingly
                    # complicated logic below does just that - for arguments (b, c, d, e)
                    # we would get a mask (1, 1, 1, 0).
                    old_args = fullargspec.args
                    old_defaults = fullargspec.defaults

                    no_default = object()
                    num_args_without_defaults = len(old_args) - len(old_defaults)
                    left_padding = tuple([no_default] * num_args_without_defaults)

                    args_with_defaults = zip(old_args, left_padding + old_defaults)

                    # Create a mask where 0 stands for args that had a partial kwarg
                    # defined.
                    non_keyword_defaults_mask = [
                        0 if key in python_function.keywords else 1 for key in old_args
                    ]
                    # Keep only arguments and defaults that were not kwargs of partial.
                    new_args_with_defaults = list(
                        itertools.compress(args_with_defaults, non_keyword_defaults_mask))
                    # Keep all args.
                    new_args = [arg for arg, _ in new_args_with_defaults]
                    # Keep only real default values.
                    new_defaults = [
                        default for _, default in new_args_with_defaults
                        if default is not no_default
                    ]
                fullargspec = inspect.FullArgSpec(
                    args=new_args,
                    varargs=fullargspec.varargs,
                    varkw=fullargspec.varkw,
                    defaults=new_defaults,
                    kwonlyargs=[],
                    kwonlydefaults={},
                    annotations=fullargspec.annotations)
        is_method = inspect.ismethod(python_function)

        # Get the function's name.  Remove functools.partial wrappers if necessary.
        while isinstance(python_function, functools.partial):
            python_function = python_function.func
        name = getattr(python_function, "__name__", "f")

        return FunctionSpec(
            fullargspec,
            is_method,
            input_signature,
            is_pure=is_pure,
            experimental_compile=experimental_compile,
            experimental_follow_type_hints=experimental_follow_type_hints,
            name=name)

    def __init__(self,
                 fullargspec,
                 is_method,
                 input_signature,
                 is_pure=False,
                 experimental_follow_type_hints=False,
                 name=None,
                 experimental_compile=None):
        """Constructs a FunctionSpec describing a python function.

    Args:
      fullargspec: `inspect.FullArgSpec` object describing the function.
      is_method: True if the function is a method.
      input_signature: a signature of the function (None, if variable)
      is_pure: if True all input arguments (including variables and constants)
        will be converted to tensors and no variable changes allowed.
      experimental_follow_type_hints: see `tf.function`.
      name: Name of the function
      experimental_compile: see `tf.function`.
    """
        self._fullargspec = fullargspec
        self._is_method = is_method
        self._is_pure = is_pure
        self._experimental_compile = experimental_compile
        self._experimental_follow_type_hints = experimental_follow_type_hints

        # TODO(edloper): Include name when serializing for SavedModel?
        self._name = name or "f"

        if self._is_method:
            # Remove `self`: default arguments shouldn't be matched to it.
            # TODO(b/127938157): Should this error out if there is no arg to
            # be removed?
            args = fullargspec.args[1:]
        else:
            args = fullargspec.args

        # A cache mapping from argument name to index, for canonicalizing
        # arguments that are called in a keyword-like fashion.
        self._args_to_indices = {arg: i for i, arg in enumerate(args)}
        self._arg_names = args

        # A cache mapping from arg index to default value, for canonicalization.
        default_values = fullargspec.defaults
        offset = len(args) - len(default_values or [])
        self._arg_indices_to_default_values = {
            offset + index: default
            for index, default in enumerate(default_values or [])
        }
        if input_signature is None:
            self._input_signature = None
        else:
            if set(fullargspec.kwonlyargs) - set(fullargspec.kwonlydefaults or ()):
                raise ValueError("Cannot define a TensorFlow function from a Python "
                                 "function with keyword-only arguments when "
                                 "input_signature is provided.")

            if not isinstance(input_signature, (tuple, list)):
                raise TypeError("input_signature must be either a tuple or a "
                                "list, received " + str(type(input_signature)))

            self._input_signature = tuple(input_signature)
            self._flat_input_signature = tuple(iteration_tools.flatten(input_signature))

    @property
    def fullargspec(self):
        """!!! note

        Failed to generate docs
        """
        return self._fullargspec

    @property
    def is_method(self):
        """!!! note

        Failed to generate docs
        """
        return self._is_method

    @property
    def args_to_indices(self):
        """!!! note

        Failed to generate docs
        """
        return self._args_to_indices

    @property
    def kwargs_to_include(self):
        """!!! note

        Failed to generate docs
        """
        return self._kwargs_to_include

    @property
    def input_signature(self):
        """!!! note

        Failed to generate docs
        """
        return self._input_signature

    @property
    def flat_input_signature(self):
        """!!! note

        Failed to generate docs
        """
        return self._flat_input_signature

    @property
    def is_pure(self):
        """!!! note

        Failed to generate docs
        """
        return self._is_pure

    @property
    def experimental_compile(self):
        """!!! note

        Failed to generate docs
        """
        return self._experimental_compile

    @property
    def arg_names(self):
        """!!! note

        Failed to generate docs
        """
        return self._arg_names

    @property
    def vararg_name(self):
        """!!! note

        Failed to generate docs
        """
        return self._fullargspec.varargs

    @property
    def varkw_name(self):
        """!!! note

        Failed to generate docs
        """
        return self._fullargspec.varkw

    def signature_summary(self, default_values=False):
        """Returns a string summarizing this function's signature.

    Args:
      default_values: If true, then include default values in the signature.

    Returns:
      A `string`.
    """
        args = list(self._arg_names)
        if default_values:
            for (i, default) in self._arg_indices_to_default_values.items():
                args[i] += "={}".format(default)
        if self._fullargspec.kwonlyargs:
            args.append("*")
            for arg_name in self._fullargspec.kwonlyargs:
                args.append(arg_name)
                if default_values and arg_name in self._fullargspec.kwonlydefaults:
                    args[-1] += "={}".format(self._fullargspec.kwonlydefaults[arg_name])
        return "{}({})".format(self._name, ", ".join(args))

    def _convert_variables_to_tensors(self, args, kwargs):
        """Converts variables to tensors.

        Args:
            args: A list of variables to be converted to tensors.
            kwargs: A dictionary of keyword arguments where the keys are variable names and the values are the variables to be converted to tensors.

        Returns:
            A tuple containing the converted variables as tensors.

        Note:
            This function uses the `to_tensor` function to convert each variable to a tensor."""
        args = [to_tensor(x) for x in args]
        kwargs = {kw: to_tensor(x) for kw, x in kwargs.items()}
        return tuple(args), kwargs

    def _convert_annotated_args_to_tensors(self, args, kwargs):
        """Attempts to autobox arguments annotated as tf.Tensor."""
        if self.input_signature is not None:
            return

        args = list(args)
        for i, arg in enumerate(args):
            # See
            # https://docs.python.org/3/library/inspect.html#inspect.getfullargspec
            if i < len(self._fullargspec.args):
                arg_annotation = self._fullargspec.annotations.get(
                    self._fullargspec.args[i])
                # TODO(rahulkamat): Change to TensorLike (here ans below).
                if arg_annotation == Tensor:
                    args[i] = to_tensor(arg)
            else:
                varargs_annotation = self._fullargspec.annotations.get(
                    self._fullargspec.varargs)
                if varargs_annotation == Tensor:
                    args[i] = to_tensor(arg)

        for kw, v in kwargs.items():
            if kw in self._fullargspec.kwonlyargs:
                kwonlyarg_annotation = self._fullargspec.annotations.get(kw)
                if kwonlyarg_annotation == Tensor:
                    kwargs[kw] = to_tensor(v)
            elif self._fullargspec.varkw is not None:
                varkw_annotation = self._fullargspec.annotations.get(
                    self._fullargspec.varkw)
                if kw in self._fullargspec.args:
                    arg_annotation = self._fullargspec.annotations.get(kw)
                    if arg_annotation == Tensor:
                        kwargs[kw] = to_tensor(v)
                elif varkw_annotation == Tensor:
                    kwargs[kw] = to_tensor(v)

        return tuple(args), kwargs

    def canonicalize_function_inputs(self, *args, **kwargs):
        """Canonicalizes `args` and `kwargs`.

    Canonicalize the inputs to the Python function using a `FunctionSpec`
    instance. In particular, we parse the varargs and kwargs that the
    original function was called with into a tuple corresponding to the
    Python function's positional (named) arguments and a dictionary
    corresponding to its kwargs.  Missing default arguments are added.

    If this `FunctionSpec` has an input signature, then it is used to convert
    arguments to tensors; otherwise, any inputs containing numpy arrays are
    converted to tensors.

    Additionally, any inputs containing numpy arrays are converted to Tensors.

    Args:
      *args: The varargs this object was called with.
      **kwargs: The keyword args this function was called with.

    Returns:
      A canonicalized ordering of the inputs, as well as full and filtered
      (Tensors and Variables only) versions of their concatenated flattened
      representations, represented by a tuple in the form (args, kwargs,
      flat_args, filtered_flat_args). Here: `args` is a full list of bound
      arguments, and `kwargs` contains only true keyword arguments, as opposed
      to named arguments called in a keyword-like fashion.

    Raises:
      ValueError: If a keyword in `kwargs` cannot be matched with a positional
        argument when an input signature is specified, or when the inputs
        do not conform to the input signature.
    """
        if self._is_pure:
            args, kwargs = self._convert_variables_to_tensors(args, kwargs)
        if self._experimental_follow_type_hints:
            args, kwargs = self._convert_annotated_args_to_tensors(args, kwargs)
        if self._input_signature is not None:
            if len(args) > len(self._input_signature):
                raise TypeError("{} takes {} positional arguments (as specified by the "
                                "input_signature) but {} were given".format(self.signature_summary(), len(self._input_signature), len(args)))
            for arg in six.iterkeys(kwargs):
                index = self._args_to_indices.get(arg, None)
                if index is None:
                    raise TypeError("{} got unexpected keyword argument `{}`".format(
                        self.signature_summary(), arg))
                if index >= len(self._input_signature):
                    raise TypeError(
                        "{} got keyword argument `{}` that was not included in "
                        "input_signature".format(self.signature_summary(), arg))

        if not kwargs:
            inputs = args
            if self._arg_indices_to_default_values:
                try:
                    inputs += tuple(
                        self._arg_indices_to_default_values[i]
                        for i in range(len(args), len(self._arg_names)))
                except KeyError:
                    missing_args = [
                        self._arg_names[i]
                        for i in range(len(args), len(self._arg_names))
                        if i not in self._arg_indices_to_default_values
                    ]
                    raise TypeError("{} missing required arguments: {}".format(
                        self.signature_summary(), ", ".join(missing_args)))

            if self._fullargspec.kwonlydefaults:
                kwargs.update(self._fullargspec.kwonlydefaults)
        else:
            # Maps from index of arg to its corresponding value, according to `args`
            # and `kwargs`; seeded with the default values for the named args that
            # aren't in `args`.
            arg_indices_to_values = {
                index: default for index, default in six.iteritems(
                    self._arg_indices_to_default_values) if index >= len(args)
            }
            consumed_args = []
            for arg, value in six.iteritems(kwargs):
                index = self._args_to_indices.get(arg, None)
                if index is not None:
                    if index < len(args):
                        raise TypeError("{} got two values for argument '{}'".format(
                            self.signature_summary(), arg))
                    arg_indices_to_values[index] = value
                    consumed_args.append(arg)
            for arg in consumed_args:
                # After this loop, `kwargs` will only contain keyword_only arguments,
                # and all positional_or_keyword arguments have been moved to `inputs`.
                kwargs.pop(arg)
            inputs = args + _deterministic_dict_values(arg_indices_to_values)

            if kwargs and self._input_signature is not None:
                raise TypeError(
                    "{} got unexpected keyword arguments: {}\n(Cannot define a "
                    "TensorFlow function from a Python function with keyword arguments "
                    "when input_signature is provided.)".format(
                        self.signature_summary(), ", ".join(kwargs)))

            if self._fullargspec.kwonlydefaults:
                for (kwarg, default) in self._fullargspec.kwonlydefaults.items():
                    kwargs.setdefault(kwarg, default)

        if self._input_signature is None:
            inputs, flat_inputs, filtered_flat_inputs = _convert_numpy_inputs(inputs)
            kwargs, flat_kwargs, filtered_flat_kwargs = _convert_numpy_inputs(kwargs)
            return (inputs, kwargs, flat_inputs + flat_kwargs,
                    filtered_flat_inputs + filtered_flat_kwargs)
        else:
            assert not kwargs
            inputs, flat_inputs, filtered_flat_inputs = _convert_inputs_to_signature(
                inputs, self._input_signature, self._flat_input_signature)
            return inputs, {}, flat_inputs, filtered_flat_inputs


#
# class FunctionCache(object):
#   """A lightweight container for cached functions.
#   """
#
#   __slots__ = [
#       "missed", "primary", "arg_relaxed_specs", "arg_relaxed",
#       "_garbage_collectors"
#   ]
#
#   def __init__(self):
#     # The set of functions that have been missed; entries are CacheKey with
#     # input_signature `None` (e.g. a "call context key")
#     self.missed = set()
#     # The primary cache, mapping a fully shaped CacheKey to a function.
#     self.primary = collections.OrderedDict()
#     # A cache key lookup, mapping a CacheKey generated without shape info to a
#     # flat list of `TypeSpec`s with relaxed shapes (one for each flattened
#     # argument). Arguments that are not Tensors or `CompositeTensor`s contain a
#     # `None` for the corresponding relaxed spec.
#     self.arg_relaxed_specs = collections.OrderedDict()
#     # The secondary cache, mapping a CacheKey generated without shape info to a
#     # function.
#     self.arg_relaxed = collections.OrderedDict()
#     # All OrderedDicts require manual garbage collection.
#     self._garbage_collectors = [
#         _FunctionGarbageCollector(self.primary),
#         _FunctionGarbageCollector(self.arg_relaxed),
#         _FunctionGarbageCollector(self.arg_relaxed_specs)]
#
#   def all_values(self):
#     """A set of all `ConcreteFunction` instances held by this cache."""
#     return set(self.primary.values()) | set(self.arg_relaxed.values())


class Function(object):
    """
    Base class of all primitive tensor operators.

    If it has only one output, one can invoke Variable methods on it, which it
    will relay to its only output.

    `Function` objects can also be constructed directly from a Python lambda,
    by means of the `@Function` decorator.
    The `Function`'s input signature is defined by the lambda.

    Example:
      >>> print(Function(to_list))    # inspect the Function's type

      ElementTimes(x: Sequence[tensor]) -> Sequence[tensor]

    The above form creates a CNTK Function whose arguments are placeholder variables.
    Such a function can only be combined with other symbolic functions.

    To train a Function or pass data to it, you need to declare the types
    of the arguments. In this case, the @Function decorator creates a CNTK Function
    whose arguments are input variables.

    ``make_block=True`` is an internal parameter used to implement :func:`@BlockFunction <cntk.ops.functions.BlockFunction>`.
    If `BlockFunction()` passes `True`, then the result will be wrapped
    in :func:`~cntk.ops.as_block()`, using the supplied ``op_name`` and ``name`` parameters, which are otherwise ignored.
    """

    # We override the constructors to implement an overload that constructs
    # a Functions from a Python function (@Function).

    def __init__(self, *args, **kwargs):
        """A class representing a Python function.

        Attributes:
            _python_function : the original Python function
            _function_spec : specification of the function
            _name : name of the function
            _autograph : flag indicating whether autograph is enabled for the function
            _autograph_options : options for autograph
            _experimental_relax_shapes : flag indicating whether shape relaxation is enabled
            _function_attributes : attributes of the function
            _capture_by_value : value used for capturing variables by value
            tracing_count : count of function tracing

        Properties:
            input_signature : input signature of the function
            flat_input_signature : flattened input signature of the function
            _hashable_input_signature : hashable input signature of the function
            _lock : lock used for thread safety
            _descriptor_cache : cache for instance-specific functions"""
        if len(args) > 0 and hasattr(args[0], '__call__') and isinstance(args[0], Function):  # overload
            return
        super(Function, self).__init__()
        self._python_function = args[0]
        self._function_spec = FunctionSpec.from_function_and_signature(
            self._python_function,
            input_signature=None,
            is_pure=None,
            experimental_follow_type_hints=False)
        self._name = kwargs.get('name', self._python_function.__qualname__)
        self._autograph = True
        self._autograph_options = None
        self._experimental_relax_shapes = False
        # self._function_cache = FunctionCache()
        self._function_attributes = {}
        self._capture_by_value = None
        self.tracing_count = 0

        if self.input_signature is not None:
            self._hashable_input_signature = self._make_input_signature_hashable(self.flat_input_signature)

        self._lock = threading.Lock()
        # _descriptor_cache is a of instance of a class to an instance-specific
        # `Function`, used to make sure defun-decorated methods create different
        # functions for each instance.
        self._descriptor_cache = weakref.WeakKeyDictionary()

    @property
    def python_function(self):
        """Returns the wrapped Python function."""
        return self._python_function  # pylint: disable=protected-access

    @property
    def function_spec(self):
        """!!! note

        Failed to generate docs
        """
        return self._function_spec

    @property
    def input_signature(self):
        """Returns the input signature."""
        return self._function_spec.input_signature

    @property
    def flat_input_signature(self):
        """Returns the flattened input signature."""
        return self._function_spec.flat_input_signature

    def _make_input_signature_hashable(self, elem):
        """Rewrite input signature to be hashable.

        We replace nested variables in the input signature with TensorSpec in order to
        be hashable.

        Args:
          elem: Input signature element

        Returns:
          A hashable object for the requested input signature
        """
        try:
            hash(elem)
        except TypeError:
            # TODO(slebedev): consider using nest.
            if isinstance(elem, tuple):
                return tuple(map(self._make_input_signature_hashable, elem))

            # TFE_Py_EncodeArg weakrefs arguments it does not recognize, and we expect
            # all recognized types to be hashable.
            assert isinstance(elem, weakref.ReferenceType)
            v = elem()

            if is_instance(v, 'tensorflow.python.ops.esourceVariable'):
                # We special case variables here to use unique_id as the cache key. This
                # ensures we have to retrace whenever a different variable is passed in.
                # This is needed to support cases where the user may use the id of a
                # variable in the function perhaps as a lookup in a dictionary.
                #
                # This choice leads to more retracing when we could have possibly used the
                # shape and dtype instead. However, we expect the number of variables in a
                # program to be bounded, and correspondingly the number of retraces.
                #
                # Note we also include the class name to avoid collisions with strings.
                return v.__class__, v._unique_id  # pylint: disable=protected-access

            if isinstance(v, np.ndarray):
                # Numpy arrays are not hashable, but when calling functions we treat them
                # in the same way as tf.Tensors.
                if not hasattr(v, "shape") or not hasattr(v, "dtype"):
                    # TODO(tomhennigan) De-dup with _as_ndarray in _convert_numpy_inputs.
                    v = to_numpy(v)
                return TensorSpec(shape=tensor_to_shape(v, need_exclude_batch_axis=True, is_singleton=True),
                                  ndim=len(v), dtype=v.dtype)

            raise ValueError("Arguments to a tf.function must be Tensors, Variables, "
                             "or hashable Python objects (or nested structures of "
                             "these types).\nGot type: %s" % type(v).__name__)

        return elem
