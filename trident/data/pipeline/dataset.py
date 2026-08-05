from __future__ import absolute_import, division, print_function

try:
    from collections.abc import Mapping
except ImportError:  # pragma: no cover - Python 3.5
    from collections import Mapping

from .batch import Sample
from .schema import DatasetSchema


def _as_sample(value, sample_id=None):
    if isinstance(value, Sample):
        sample = value.copy()
        if sample.sample_id is None:
            sample.sample_id = sample_id
        return sample
    if isinstance(value, Mapping):
        return Sample(value, sample_id=sample_id)
    if isinstance(value, tuple):
        return Sample((("field_{0}".format(index), item)
                       for index, item in enumerate(value)), sample_id=sample_id)
    return Sample({"data": value}, sample_id=sample_id)


def _columnar_to_samples(batch):
    if not isinstance(batch, Mapping):
        return [_as_sample(item) for item in batch]
    if not batch:
        return []
    lengths = [len(value) for value in batch.values()]
    if len(set(lengths)) != 1:
        raise ValueError("columnar batch fields have different lengths")
    return [Sample((key, value[index]) for key, value in batch.items())
            for index in range(lengths[0])]


class Dataset(object):
    """Framework-neutral view over a map-style or iterable data source."""

    def __init__(self, source, schema=None, map_style=None, name=None,
                 validate_samples=True):
        self.source = source
        if schema is None:
            schema = DatasetSchema()
        elif isinstance(schema, Mapping):
            schema = DatasetSchema.from_dict(schema)
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema or its dictionary representation")
        self.schema = schema
        self.validate_samples = bool(validate_samples)
        self.name = name or getattr(source, "name", source.__class__.__name__)
        if map_style is None:
            map_style = hasattr(source, "__len__") and hasattr(source, "__getitem__")
        self.map_style = bool(map_style)
        if not self.map_style and not hasattr(source, "__iter__"):
            raise TypeError("iterable dataset source must implement __iter__")

    @classmethod
    def from_columns(cls, columns, schema=None, name=None, validate_samples=True):
        from .adapters import ColumnarSource
        return cls(ColumnarSource(columns), schema=schema, map_style=True,
                   name=name or "columns", validate_samples=validate_samples)

    @classmethod
    def from_iterable(cls, factory, schema=None, name=None, validate_samples=True):
        from .adapters import IterableFactorySource
        return cls(IterableFactorySource(factory), schema=schema, map_style=False,
                   name=name or "iterable", validate_samples=validate_samples)

    @classmethod
    def from_folder(cls, root, patterns=None, recursive=True, loader=None,
                    field="data", path_field="path", label_from_parent=False,
                    label_field="label", schema=None, name=None,
                    validate_samples=True):
        from .adapters import FolderSource
        source = FolderSource(
            root, patterns=patterns, recursive=recursive, loader=loader,
            field=field, path_field=path_field,
            label_from_parent=label_from_parent, label_field=label_field)
        return cls(source, schema=schema, map_style=True,
                   name=name or source.root.name, validate_samples=validate_samples)

    @classmethod
    def from_legacy(cls, data=None, label=None, unpair=None, schema=None,
                    name=None, validate_samples=True):
        from .adapters import LegacySource
        return cls(LegacySource(data=data, label=label, unpair=unpair),
                   schema=schema, map_style=True, name=name or "legacy",
                   validate_samples=validate_samples)

    @classmethod
    def from_huggingface(cls, source, schema=None, name=None, validate_samples=True):
        """Wrap a Hugging Face Dataset or IterableDataset without materializing it."""
        module_name = source.__class__.__module__
        if not module_name.startswith("datasets") and not (
                hasattr(source, "features") and hasattr(source, "column_names")):
            raise TypeError("source does not look like a Hugging Face dataset")
        map_style = hasattr(source, "__len__") and hasattr(source, "__getitem__")
        return HuggingFaceDataset(source, schema=schema, map_style=map_style,
                                  name=name, validate_samples=validate_samples)

    @property
    def iterable_style(self):
        return not self.map_style

    def __len__(self):
        if not self.map_style:
            raise TypeError("iterable-style dataset has no length")
        return len(self.source)

    def _prepare(self, value, sample_id=None):
        sample = _as_sample(value, sample_id=sample_id)
        if self.validate_samples and len(self.schema):
            self.schema.validate(sample)
        return sample

    def __getitem__(self, index):
        if not self.map_style:
            raise TypeError("iterable-style dataset does not support indexing")
        return self._prepare(self.source[index], sample_id=index)

    def __iter__(self):
        if self.map_style:
            for index in range(len(self)):
                yield self[index]
        else:
            for index, value in enumerate(self.source):
                yield self._prepare(value, sample_id=index)

    def get_batch(self, indices):
        if not self.map_style:
            raise TypeError("iterable-style dataset does not support batched indexing")
        getitems = getattr(self.source, "__getitems__", None)
        if callable(getitems):
            values = getitems(list(indices))
            return [self._prepare(value, sample_id=index)
                    for index, value in zip(indices, values)]
        return [self[index] for index in indices]

    def with_transform(self, transform):
        return TransformedDataset(self, transform)

    def select_columns(self, columns):
        return ProjectedDataset(self, columns)

    def set_epoch(self, epoch):
        method = getattr(self.source, "set_epoch", None)
        if callable(method):
            method(int(epoch))
        return self

    def shuffle(self, seed=0, buffer_size=None):
        method = getattr(self.source, "shuffle", None)
        if not callable(method):
            if self.map_style:
                raise TypeError("map-style shuffling belongs to Iterator(shuffle=True)")
            raise TypeError("iterable source does not provide a shuffle buffer")
        kwargs = {"seed": int(seed)}
        if buffer_size is not None:
            kwargs["buffer_size"] = int(buffer_size)
        shuffled = method(**kwargs)
        return self.__class__(shuffled, schema=self.schema,
                              map_style=self.map_style, name=self.name,
                              validate_samples=self.validate_samples)

    def shard(self, num_shards, index):
        if num_shards <= 0 or not 0 <= index < num_shards:
            raise ValueError("invalid shard {0}/{1}".format(index, num_shards))
        source_shard = getattr(self.source, "shard", None)
        if callable(source_shard):
            try:
                return self.__class__(source_shard(num_shards=num_shards, index=index),
                                      schema=self.schema, map_style=self.map_style,
                                      name=self.name,
                                      validate_samples=self.validate_samples)
            except TypeError:
                pass
        return ShardedDataset(self, num_shards, index)


class HuggingFaceDataset(Dataset):
    def with_format(self, format_type=None, columns=None, output_all_columns=False,
                    **format_kwargs):
        method = getattr(self.source, "with_format", None)
        if not callable(method):
            raise TypeError("Hugging Face source does not provide with_format")
        formatted = method(
            type=format_type, columns=columns,
            output_all_columns=output_all_columns, **format_kwargs)
        return HuggingFaceDataset(
            formatted, schema=self.schema, map_style=self.map_style,
            name=self.name, validate_samples=self.validate_samples)

    def with_huggingface_transform(self, transform, columns=None,
                                   output_all_columns=False):
        method = getattr(self.source, "with_transform", None)
        if not callable(method):
            raise TypeError("Hugging Face source does not provide with_transform")
        transformed = method(
            transform, columns=columns,
            output_all_columns=output_all_columns)
        return HuggingFaceDataset(
            transformed, schema=self.schema, map_style=self.map_style,
            name=self.name, validate_samples=self.validate_samples)

    def select_columns(self, columns):
        method = getattr(self.source, "select_columns", None)
        if callable(method):
            selected = method(columns)
            schema = DatasetSchema([self.schema[name] for name in columns
                                    if name in self.schema])
            return HuggingFaceDataset(
                selected, schema=schema, map_style=self.map_style,
                name=self.name, validate_samples=self.validate_samples)
        return Dataset.select_columns(self, columns)

    def get_batch(self, indices):
        if not self.map_style:
            return Dataset.get_batch(self, indices)
        values = _columnar_to_samples(self.source[list(indices)])
        return [self._prepare(value, sample_id=index)
                for index, value in zip(indices, values)]


class ProjectedDataset(Dataset):
    def __init__(self, dataset, columns):
        self.dataset = dataset
        self.columns = tuple(columns)
        missing = [name for name in self.columns
                   if len(dataset.schema) and name not in dataset.schema]
        if missing:
            raise KeyError("columns not present in schema: {0}".format(missing))
        schema = DatasetSchema([dataset.schema[name] for name in self.columns
                                if name in dataset.schema])
        Dataset.__init__(self, dataset, schema=schema,
                         map_style=dataset.map_style,
                         name="{0}.columns".format(dataset.name),
                         validate_samples=False)

    def _project(self, sample):
        missing = [name for name in self.columns if name not in sample]
        if missing:
            raise KeyError("sample is missing selected columns: {0}".format(missing))
        return Sample(((name, sample[name]) for name in self.columns),
                      sample_id=getattr(sample, "sample_id", None),
                      metadata=getattr(sample, "metadata", None))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._project(self.dataset[index])

    def __iter__(self):
        for sample in self.dataset:
            yield self._project(sample)

    def get_batch(self, indices):
        return [self._project(sample) for sample in self.dataset.get_batch(indices)]


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        Dataset.__init__(self, dataset, schema=dataset.schema,
                         map_style=dataset.map_style,
                         name="{0}.transform".format(dataset.name),
                         validate_samples=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset[index])

    def __iter__(self):
        for sample in self.dataset:
            yield self.transform(sample)

    def get_batch(self, indices):
        return [self.transform(sample) for sample in self.dataset.get_batch(indices)]


class ShardedDataset(Dataset):
    def __init__(self, dataset, num_shards, index):
        self.dataset = dataset
        self.num_shards = num_shards
        self.index = index
        Dataset.__init__(self, dataset, schema=dataset.schema,
                         map_style=dataset.map_style,
                         name="{0}.shard{1}".format(dataset.name, index),
                         validate_samples=False)

    def __len__(self):
        total = len(self.dataset)
        if total <= self.index:
            return 0
        return ((total - self.index - 1) // self.num_shards) + 1

    def __getitem__(self, index):
        return self.dataset[self.index + index * self.num_shards]

    def __iter__(self):
        for position, sample in enumerate(self.dataset):
            if position % self.num_shards == self.index:
                yield sample
