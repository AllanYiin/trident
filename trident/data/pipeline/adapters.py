from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from pathlib import Path

try:
    from collections.abc import Mapping
except ImportError:  # pragma: no cover - Python 3.5
    from collections import Mapping


class ColumnarSource(object):
    """Map-style source over equal-length columns without copying them."""

    def __init__(self, columns):
        if not isinstance(columns, Mapping) or not columns:
            raise ValueError("columns must be a non-empty mapping")
        self.columns = OrderedDict(columns)
        lengths = [len(value) for value in self.columns.values()]
        if len(set(lengths)) != 1:
            raise ValueError("column lengths must match")
        self.length = lengths[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return dict((name, values[index]) for name, values in self.columns.items())

    def __getitems__(self, indices):
        return [self[index] for index in indices]


class IterableFactorySource(object):
    """Re-iterable source that creates a fresh iterator for every epoch."""

    def __init__(self, factory):
        if not callable(factory):
            raise TypeError("iterable factory must be callable")
        self.factory = factory

    def __iter__(self):
        source = self.factory()
        if not hasattr(source, "__iter__"):
            raise TypeError("iterable factory must return an iterable")
        return iter(source)


class FolderSource(object):
    """Lazy file source; enumeration stores paths while loading stays per-sample."""

    def __init__(self, root, patterns=None, recursive=True, loader=None,
                 field="data", path_field="path", label_from_parent=False,
                 label_field="label"):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError("folder does not exist: {0}".format(self.root))
        if patterns is None:
            patterns = ("*",)
        elif isinstance(patterns, str):
            patterns = (patterns,)
        self.patterns = tuple(patterns)
        self.recursive = bool(recursive)
        self.loader = loader
        self.field = field
        self.path_field = path_field
        self.label_from_parent = bool(label_from_parent)
        self.label_field = label_field
        paths = []
        for pattern in self.patterns:
            matches = self.root.rglob(pattern) if self.recursive else self.root.glob(pattern)
            paths.extend(path for path in matches if path.is_file())
        self.paths = tuple(sorted(set(paths), key=lambda value: str(value).lower()))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        path_text = str(path)
        value = self.loader(path_text) if self.loader is not None else path_text
        sample = {self.field: value}
        if self.path_field and self.path_field != self.field:
            sample[self.path_field] = path_text
        if self.label_from_parent:
            sample[self.label_field] = path.parent.name
        return sample

    def __getitems__(self, indices):
        return [self[index] for index in indices]


def _legacy_symbols(source, fallback):
    symbols = getattr(source, "symbol", fallback)
    if isinstance(symbols, tuple):
        return list(symbols)
    if isinstance(symbols, list):
        return symbols
    return [symbols or fallback]


class LegacySource(object):
    """Lazy adapter over legacy data/label/unpair datasets."""

    def __init__(self, data=None, label=None, unpair=None):
        groups = []
        for fallback, source in (("data", data), ("label", label), ("unpair", unpair)):
            if source is not None and len(source) > 0:
                groups.append((source, _legacy_symbols(source, fallback)))
        if not groups:
            raise ValueError("at least one non-empty legacy dataset is required")
        self.groups = tuple(groups)
        self.length = max(len(source) for source, _ in groups)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample = OrderedDict()
        for source, symbols in self.groups:
            value = source[index % len(source)]
            values = value if isinstance(value, tuple) else (value,)
            if len(values) != len(symbols):
                raise ValueError("legacy symbols and returned values do not match")
            for symbol, item in zip(symbols, values):
                if symbol in sample:
                    raise ValueError("duplicate legacy symbol: {0}".format(symbol))
                sample[symbol] = item
        return sample

    def __getitems__(self, indices):
        return [self[index] for index in indices]
