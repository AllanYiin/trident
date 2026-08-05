from __future__ import absolute_import, division, print_function

import hashlib
import json
from collections import OrderedDict

import numpy as np

try:
    from collections.abc import Mapping
except ImportError:  # pragma: no cover - Python 3.5
    from collections import Mapping

from .errors import SchemaValidationError


_SPATIAL_KINDS = frozenset((
    "image", "mask", "depth", "bbox", "keypoints", "landmarks", "polygon",
    "densepose", "optical_flow",
))


class FieldSpec(object):
    """Serializable description of a sample field without owning its value."""

    def __init__(self, name, kind="data", dtype=None, shape=None, layout=None,
                 variable_axes=None, pad_value=None, coordinate_format=None,
                 interpolation=None, required=True, metadata=None):
        if not isinstance(name, str) or not name:
            raise ValueError("FieldSpec.name must be a non-empty string")
        self.name = name
        self.kind = kind or "data"
        self.dtype = dtype
        self.shape = tuple(shape) if shape is not None else None
        self.layout = layout
        self.variable_axes = tuple(sorted(set(variable_axes or ())))
        if any(not isinstance(axis, int) or axis < 0 for axis in self.variable_axes):
            raise ValueError("variable_axes must contain non-negative integers")
        if self.shape is not None and any(axis >= len(self.shape) for axis in self.variable_axes):
            raise ValueError("variable axis is outside FieldSpec.shape")
        self.pad_value = pad_value
        self.coordinate_format = coordinate_format
        self.interpolation = interpolation
        self.required = bool(required)
        self.metadata = dict(metadata or {})

    @property
    def is_spatial(self):
        return self.kind in _SPATIAL_KINDS

    @property
    def role(self):
        return self.metadata.get("role")

    def copy(self, **changes):
        values = self.to_dict()
        values["dtype"] = self.dtype
        values.update(changes)
        return FieldSpec(**values)

    def validate_value(self, value):
        if value is None:
            if self.required:
                raise SchemaValidationError(
                    "required field cannot be None", field=self.name)
            return value
        actual_shape = getattr(value, "shape", None)
        if self.shape is not None:
            if actual_shape is None:
                actual_shape = np.shape(value)
            actual_shape = tuple(actual_shape)
            if len(actual_shape) != len(self.shape):
                raise SchemaValidationError(
                    "rank mismatch: expected {0}, got {1}".format(
                        len(self.shape), len(actual_shape)), field=self.name)
            for axis, (expected, actual) in enumerate(zip(self.shape, actual_shape)):
                if axis in self.variable_axes or expected is None:
                    continue
                if int(expected) != int(actual):
                    raise SchemaValidationError(
                        "shape mismatch at axis {0}: expected {1}, got {2}".format(
                            axis, expected, actual), field=self.name)
        if self.dtype is not None:
            if self.dtype in (str, "str", "string", "unicode"):
                valid = isinstance(value, str) or (
                    hasattr(value, "dtype") and value.dtype.kind in ("U", "S"))
            else:
                actual_dtype = getattr(value, "dtype", None)
                if actual_dtype is None:
                    try:
                        actual_dtype = np.asarray(value).dtype
                    except Exception:
                        actual_dtype = None
                try:
                    valid = actual_dtype is not None and np.dtype(actual_dtype) == np.dtype(self.dtype)
                except TypeError:
                    valid = str(actual_dtype) == str(self.dtype)
            if not valid:
                raise SchemaValidationError(
                    "dtype mismatch: expected {0}, got {1}".format(
                        self.dtype, getattr(value, "dtype", type(value).__name__)),
                    field=self.name)
        return value

    def to_dict(self):
        dtype = self.dtype
        if dtype is not None and dtype is not str:
            try:
                dtype = str(np.dtype(dtype))
            except TypeError:
                dtype = str(dtype)
        elif dtype is str:
            dtype = "str"
        return dict(
            name=self.name, kind=self.kind, dtype=dtype,
            shape=list(self.shape) if self.shape is not None else None,
            layout=self.layout, variable_axes=list(self.variable_axes),
            pad_value=self.pad_value, coordinate_format=self.coordinate_format,
            interpolation=self.interpolation, required=self.required,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_dict(cls, values):
        return cls(**dict(values))

    def __eq__(self, other):
        return isinstance(other, FieldSpec) and self.to_dict() == other.to_dict()

    def __repr__(self):
        return "FieldSpec(name={0!r}, kind={1!r}, shape={2!r})".format(
            self.name, self.kind, self.shape)


class DatasetSchema(object):
    """Ordered, serializable collection of :class:`FieldSpec` objects."""

    def __init__(self, fields=None):
        self._fields = OrderedDict()
        if fields is not None:
            source = fields.values() if isinstance(fields, Mapping) else fields
            for field in source:
                self.add(field)

    def add(self, field):
        if not isinstance(field, FieldSpec):
            raise TypeError("schema fields must be FieldSpec instances")
        if field.name in self._fields:
            raise ValueError("duplicate field: {0}".format(field.name))
        self._fields[field.name] = field
        return self

    def get(self, name, default=None):
        return self._fields.get(name, default)

    def spatial_fields(self):
        return [field for field in self._fields.values() if field.is_spatial]

    def fields_by_role(self, role):
        return [field for field in self._fields.values() if field.role == role]

    def validate(self, sample, strict=False):
        if not isinstance(sample, Mapping):
            raise SchemaValidationError("sample must be a mapping")
        missing = [name for name, field in self._fields.items()
                   if field.required and name not in sample]
        if missing:
            raise SchemaValidationError(
                "sample is missing required fields: {0}".format(missing))
        if strict:
            extras = [name for name in sample if name not in self._fields]
            if extras:
                raise SchemaValidationError(
                    "sample contains fields not present in schema: {0}".format(extras))
        for name, field in self._fields.items():
            if name in sample:
                field.validate_value(sample[name])
        return sample

    def project(self, sample, include_optional=True):
        self.validate(sample)
        return OrderedDict((name, sample[name]) for name, field in self._fields.items()
                           if name in sample and (include_optional or field.required))

    def to_dict(self):
        return {"fields": [field.to_dict() for field in self._fields.values()]}

    @classmethod
    def from_dict(cls, values):
        return cls(FieldSpec.from_dict(field) for field in values.get("fields", ()))

    @property
    def fingerprint(self):
        payload = json.dumps(self.to_dict(), ensure_ascii=True,
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __getitem__(self, name):
        return self._fields[name]

    def __contains__(self, name):
        return name in self._fields

    def __iter__(self):
        return iter(self._fields.values())

    def __len__(self):
        return len(self._fields)

    def keys(self):
        return self._fields.keys()

    def values(self):
        return self._fields.values()

    def items(self):
        return self._fields.items()

    def __eq__(self, other):
        return isinstance(other, DatasetSchema) and self.to_dict() == other.to_dict()

    def __repr__(self):
        return "DatasetSchema({0!r})".format(list(self._fields.values()))


def infer_field_kind(name):
    """Conservative name-based fallback used only when no schema is supplied."""
    lowered = name.lower()
    if lowered in ("image", "images", "img", "input_image"):
        return "image"
    if "mask" in lowered:
        return "mask"
    if "depth" in lowered:
        return "depth"
    if "bbox" in lowered or "boxes" in lowered:
        return "bbox"
    if "landmark" in lowered:
        return "landmarks"
    if "keypoint" in lowered:
        return "keypoints"
    if "polygon" in lowered:
        return "polygon"
    if lowered in ("text", "sentence", "corpus", "prompt"):
        return "text"
    if lowered in ("label", "labels", "target", "targets"):
        return "label"
    return "data"