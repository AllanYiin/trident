from __future__ import absolute_import, division, print_function

try:
    from collections.abc import Mapping
except ImportError:  # pragma: no cover - Python 3.5
    from collections import Mapping


def _map_values(value, operation):
    if isinstance(value, Mapping):
        return dict((key, _map_values(item, operation))
                    for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_map_values(item, operation) for item in value)
    if isinstance(value, list):
        return [_map_values(item, operation) for item in value]
    return operation(value)


class Sample(dict):
    """Native mapping sample with optional identity and lightweight metadata."""

    def __init__(self, values=None, sample_id=None, metadata=None):
        dict.__init__(self, values or {})
        self.sample_id = sample_id
        self.metadata = dict(metadata or {})

    def copy(self):
        return Sample(self, sample_id=self.sample_id, metadata=self.metadata)


class Batch(dict):
    """Mapping batch with optional schema and backend transfer helpers."""

    def __init__(self, values=None, schema=None, metadata=None):
        dict.__init__(self, values or {})
        self.schema = schema
        self.metadata = dict(metadata or {})

    def copy(self):
        return Batch(self, schema=self.schema, metadata=self.metadata)

    def pin_memory(self):
        def pin(value):
            method = getattr(value, "pin_memory", None)
            return method() if callable(method) else value
        pinned = _map_values(self, pin)
        return Batch(pinned, schema=self.schema, metadata=self.metadata)

    def to(self, device=None, non_blocking=False):
        def transfer(value):
            method = getattr(value, "to", None)
            if not callable(method):
                return value
            try:
                return method(device, non_blocking=non_blocking)
            except TypeError:
                return method(device)
        moved = _map_values(self, transfer)
        return Batch(moved, schema=self.schema, metadata=self.metadata)