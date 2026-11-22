"""Bind callable parameters to existing flat training fields."""
import inspect
from collections.abc import Mapping
from functools import update_wrapper

from trident.backend.common import OrderedDict
from trident.backend.tensorspec import TensorSpec, get_signature


class FunctionObjective:
    """Callable adapter. Each trainer registration gets its own bindings and options."""

    def __init__(self, fn, *, input_names=None, name=None, options=None):
        update_wrapper(self, fn, updated=())
        self.__name__ = name or getattr(fn, '__name__', type(fn).__name__)
        self._signature = get_signature(fn, self.__name__)
        self._parameters = inspect.signature(fn).parameters
        self.options = dict(options or {})
        self.input_names = normalize_names(input_names, self._parameters)
        for key in self.options:
            if key not in self._parameters or key in self.input_names:
                raise ValueError(f"{self.__name__}: invalid or multiply bound option {key!r}")
        self.reduction = getattr(fn, 'reduction', 'mean')

    @property
    def signature(self):
        return self._signature

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **{**self.options, **kwargs})

    def call_fields(self, values):
        # Positional-only slots may precede a supplied optional slot.
        bound = inspect.BoundArguments(inspect.signature(self.__wrapped__), dict(values, **self.options))
        bound.apply_defaults()
        return self.__wrapped__(*bound.args, **bound.kwargs)


def normalize_names(names, parameters):
    if names is None:
        return {}
    if isinstance(names, (tuple, list)):
        if len(names) != len(parameters):
            raise ValueError("input_names must match the number of parameters")
        names = dict(zip(parameters, names))
    if not isinstance(names, Mapping):
        raise TypeError("input_names must map parameter names to field names")
    for key, value in names.items():
        if key not in parameters or not isinstance(value, str) or not value:
            raise ValueError(f"invalid input_names binding {key!r} -> {value!r}; parameters: {list(parameters)}")
    return dict(names)


def prepare_objective(obj, name, names, options, registry, lookup, adapter):
    if isinstance(obj, str):
        name = name or obj
        obj = lookup(obj)
    if inspect.isclass(obj):
        obj = obj(**options)
        options = {}
    if not callable(obj):
        raise TypeError("loss or metric must be callable")
    alias = name or getattr(obj, '__name__', type(obj).__name__)
    base = alias
    suffix = 1
    while alias in registry:
        alias = f"{base}_{suffix}"
        suffix += 1
    if isinstance(obj, FunctionObjective):
        bindings = {**obj.input_names, **normalize_names(names, obj._parameters)}
        obj = adapter(obj.__wrapped__, name=alias, input_names=bindings, options={**obj.options, **options})
    elif inspect.isroutine(obj) or not hasattr(obj, 'signature'):
        obj = adapter(obj, name=alias, input_names=names, options=options)
    elif options:
        raise TypeError(f"{alias}: configure an instantiated objective at construction")
    bindings = obj.input_names if isinstance(obj, FunctionObjective) else normalize_names(names, obj.signature.inputs)
    return alias, obj, bindings


def shape_problem(spec, value):
    if spec is None or value is None:
        return
    shape = getattr(value, 'shape', None)
    if shape is None or getattr(shape, 'rank', 0) is None:
        return
    actual = tuple(shape)
    expected = getattr(spec, 'shape', None)
    if expected is not None:
        expected = tuple(expected.dims)
        # get_signature uses [None] as a placeholder for a bare Tensor annotation.
        if getattr(spec, '_generic_tensor', False) and expected == (None,):
            return
        if len(expected) != len(actual) or any(
            a is not None and e is not None and e != a for e, a in zip(expected, actual)
        ):
            return f"expected shape {expected}, received {actual}"
    elif spec.ndim is not None and len(actual) != spec.ndim:
        return f"expected rank {spec.ndim}, received shape {actual}"
    if ((spec.min_ndim is not None and len(actual) < spec.min_ndim)
            or (spec.max_ndim is not None and len(actual) > spec.max_ndim)):
        return f"shape {actual} outside declared rank range {spec.min_ndim}..{spec.max_ndim}"
    for axis, size in (spec.axes or {}).items():
        if not -len(actual) <= axis < len(actual) or size is not None and actual[axis] not in (None, size):
            return f"expected axis {axis} of size {size}, received {actual}"


def resolve_fields(signature, data, feed, *, owner, model_inputs=(), outputs=(), labels=(),
                   local=None, model=False, fixed=()):
    resolved = OrderedDict()
    first_batch = None
    for arg, spec in signature.inputs.items():
        if arg in fixed:
            continue
        explicit = (local or {}).get(arg) or feed.get(arg)
        optional = getattr(spec, 'optional', False)
        hint = 'data_feed' if model else 'input_names'
        if explicit:
            field = explicit
        elif optional:
            continue
        elif arg in data:
            field = arg
        else:
            if model or arg in ('x', 'input'):
                candidates = model_inputs
            elif arg in ('output', 'y_pred', 'y'):
                candidates = outputs
            elif arg in ('target', 'y_true', 'label'):
                candidates = labels
            else:
                candidates = ()
            candidates = sorted({k for k in candidates if k in data and not shape_problem(spec, data[k])})
            if len(candidates) != 1:
                example = candidates[0] if candidates else '<field>'
                raise ValueError(f"{owner}.{arg}: candidates {candidates}; available fields {list(data)}. "
                                 f"Specify {hint}={{{arg!r}: {example!r}}}")
            field = candidates[0]
        if field not in data:
            raise ValueError(f"{owner}.{arg}: field {field!r} missing; available fields {list(data)}. "
                             f"Check {hint}[{arg!r}]")
        issue = shape_problem(spec, data[field])
        if issue:
            raise ValueError(f"{owner}.{arg} <- {field!r}: {issue}")
        shape = getattr(data[field], 'shape', None)
        if (model or field in labels or field in outputs or field in model_inputs) and shape is not None and len(shape):
            if first_batch is None:
                first_batch = (field, shape[0])
            elif shape[0] is not None and first_batch[1] is not None and shape[0] != first_batch[1]:
                raise ValueError(f"{owner}.{arg} <- {field!r}: batch dimension {shape[0]} "
                                 f"differs from {first_batch[0]!r} ({first_batch[1]})")
        resolved[arg] = field
    return resolved
