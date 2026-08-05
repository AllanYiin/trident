from __future__ import absolute_import, division, print_function

import numbers

import numpy as np

try:
    from collections.abc import Mapping
except ImportError:  # pragma: no cover - Python 3.5
    from collections import Mapping

from .batch import Batch


def _uniform_numeric_sequence(values):
    if not values or not all(isinstance(value, (list, tuple)) for value in values):
        return False
    lengths = [len(value) for value in values]
    if len(set(lengths)) != 1:
        return False
    return all(all(isinstance(item, numbers.Number) for item in value)
               for value in values)


def collate_values(values):
    first = values[0]
    if isinstance(first, np.ndarray):
        if all(value.shape == first.shape and value.dtype != object for value in values):
            return np.stack(values, axis=0)
        return list(values)
    if isinstance(first, numbers.Number) or isinstance(first, np.generic):
        return np.asarray(values)
    if isinstance(first, str) or isinstance(first, bytes):
        return list(values)
    if isinstance(first, Mapping):
        keys = first.keys()
        if not all(set(value.keys()) == set(keys) for value in values):
            return list(values)
        return dict((key, collate_values([value[key] for value in values]))
                    for key in keys)
    if _uniform_numeric_sequence(values):
        return np.asarray(values)
    return list(values)


def default_collate(samples, schema=None):
    if not samples:
        return Batch(schema=schema)
    keys = []
    seen = set()
    for sample in samples:
        for key in sample:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    values = {}
    for key in keys:
        field_values = [sample.get(key) for sample in samples]
        if any(value is None for value in field_values):
            values[key] = field_values
        else:
            values[key] = collate_values(field_values)
    return Batch(values, schema=schema)


default_collate.accepts_schema = True


def _pad_arrays(values, pad_value=0, axis=0, pad_to_multiple_of=None):
    arrays = [np.asarray(value) for value in values]
    max_length = max(array.shape[axis] for array in arrays)
    if pad_to_multiple_of:
        multiple = int(pad_to_multiple_of)
        max_length = ((max_length + multiple - 1) // multiple) * multiple
    output_shape = [len(arrays)] + list(arrays[0].shape)
    output_shape[axis + 1] = max_length
    output = np.full(output_shape, pad_value, dtype=arrays[0].dtype)
    lengths = []
    for row, array in enumerate(arrays):
        length = array.shape[axis]
        lengths.append(length)
        slices = [slice(None)] * output.ndim
        slices[0] = row
        slices[axis + 1] = slice(0, length)
        output[tuple(slices)] = array
    return output, np.asarray(lengths, dtype=np.int64)


class PaddingCollator(object):
    accepts_schema = True

    def __init__(self, fields=None, pad_to_multiple_of=None, return_lengths=True):
        self.fields = dict(fields or {})
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_lengths = bool(return_lengths)

    def __call__(self, samples, schema=None):
        batch = default_collate(samples, schema=schema)
        for name, options in self.fields.items():
            values = [sample[name] for sample in samples]
            field = schema.get(name) if schema is not None else None
            return_mask = False
            if isinstance(options, dict):
                pad_value = options.get("pad_value", getattr(field, "pad_value", 0) or 0)
                axis = options.get("axis", 0)
                return_mask = bool(options.get("return_mask", False))
            else:
                pad_value = options
                axis = 0
            padded, lengths = _pad_arrays(
                values, pad_value=pad_value, axis=axis,
                pad_to_multiple_of=self.pad_to_multiple_of)
            batch[name] = padded
            if self.return_lengths:
                batch["{0}_lengths".format(name)] = lengths
            if return_mask:
                batch["{0}_mask".format(name)] = (
                    np.arange(padded.shape[axis + 1])[None, :] < lengths[:, None])
        return batch


def _normalize_tokenizer_output(encoded):
    output = {}
    for key, value in encoded.items():
        if isinstance(value, np.ndarray) or hasattr(value, "device"):
            output[key] = value
        elif isinstance(value, (list, tuple)) and value and all(
                isinstance(item, (list, tuple)) for item in value):
            lengths = [len(item) for item in value]
            output[key] = np.asarray(value) if len(set(lengths)) == 1 else list(value)
        elif isinstance(value, (list, tuple)) and all(
                isinstance(item, numbers.Number) for item in value):
            output[key] = np.asarray(value)
        else:
            output[key] = value
    return output


class TokenizerCollator(object):
    """Batched tokenizer call followed by collation of non-text fields."""

    accepts_schema = True

    def __init__(self, tokenizer, text_field="text", keep_text=False,
                 tokenizer_kwargs=None, **kwargs):
        if not callable(tokenizer):
            raise TypeError("tokenizer must be callable")
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.keep_text = bool(keep_text)
        self.tokenizer_kwargs = dict(tokenizer_kwargs or {})
        self.tokenizer_kwargs.update(kwargs)
        self.tokenizer_kwargs.setdefault("padding", "longest")
        self.tokenizer_kwargs.setdefault("return_tensors", None)

    def __call__(self, samples, schema=None):
        texts = [sample[self.text_field] for sample in samples]
        encoded = _normalize_tokenizer_output(
            self.tokenizer(texts, **self.tokenizer_kwargs))
        remaining = []
        for sample in samples:
            values = dict(sample)
            if not self.keep_text:
                values.pop(self.text_field, None)
            remaining.append(values)
        batch = default_collate(remaining, schema=schema)
        batch.update(encoded)
        return batch


class HuggingFaceCollator(object):
    accepts_schema = True

    def __init__(self, collator):
        if not callable(collator):
            raise TypeError("collator must be callable")
        self.collator = collator

    def __call__(self, samples, schema=None):
        values = self.collator(samples)
        return values if isinstance(values, Batch) else Batch(values, schema=schema)



class RaggedBatch(object):
    """Compact first-axis ragged representation without NumPy object arrays."""

    def __init__(self, values, row_splits):
        self.values = values
        self.row_splits = (row_splits if hasattr(row_splits, "device") else
                           np.asarray(row_splits, dtype=np.int64))
        if (self.row_splits.ndim != 1 or len(self.row_splits) == 0 or
                int(self.row_splits[0]) != 0):
            raise ValueError("row_splits must be a 1-D array starting at zero")
        if int(self.row_splits[-1]) != len(values):
            raise ValueError("last row split must equal flattened value count")

    @property
    def nbytes(self):
        return int(getattr(self.values, "nbytes", 0) +
                   getattr(self.row_splits, "nbytes", 0))

    def pin_memory(self):
        values = self.values.pin_memory() if hasattr(self.values, "pin_memory") else self.values
        splits = (self.row_splits.pin_memory()
                  if hasattr(self.row_splits, "pin_memory") else self.row_splits)
        return RaggedBatch(values, splits)

    def to(self, device, non_blocking=True):
        def move(value):
            if not hasattr(value, "to"):
                return value
            try:
                return value.to(device, non_blocking=non_blocking)
            except TypeError:
                return value.to(device)
        return RaggedBatch(move(self.values), move(self.row_splits))

    @classmethod
    def from_sequences(cls, sequences):
        arrays = [np.asarray(sequence) for sequence in sequences]
        if not arrays:
            return cls(np.asarray([]), [0])
        trailing = arrays[0].shape[1:]
        if any(array.shape[1:] != trailing for array in arrays):
            raise ValueError("ragged sequences may vary only on the first axis")
        row_splits = np.concatenate(([0], np.cumsum([len(array) for array in arrays])))
        values = (np.concatenate(arrays, axis=0) if row_splits[-1]
                  else np.empty((0,) + trailing, dtype=arrays[0].dtype))
        return cls(values, row_splits)

    @property
    def lengths(self):
        return np.diff(self.row_splits)

    def to_list(self):
        return [self.values[self.row_splits[index]:self.row_splits[index + 1]]
                for index in range(len(self.row_splits) - 1)]

    def to_padded(self, pad_value=0, pad_to_multiple_of=None):
        rows = self.to_list()
        if not rows:
            return np.asarray([]), np.asarray([], dtype=bool)
        padded, _ = _pad_arrays(rows, pad_value=pad_value,
                                pad_to_multiple_of=pad_to_multiple_of)
        mask = np.arange(padded.shape[1])[None, :] < self.lengths[:, None]
        return padded, mask


class RaggedCollator(object):
    """Collate configured fields into values + row_splits storage."""

    accepts_schema = True

    def __init__(self, fields):
        self.fields = tuple(fields)

    def __call__(self, samples, schema=None):
        batch = default_collate(samples, schema=schema)
        for name in self.fields:
            batch[name] = RaggedBatch.from_sequences([sample[name] for sample in samples])
        return batch


class FieldCollator(object):
    """Apply field-specific value collators after conservative default collation."""

    accepts_schema = True

    def __init__(self, fields):
        self.fields = dict(fields)

    def __call__(self, samples, schema=None):
        batch = default_collate(samples, schema=schema)
        for name, collator in self.fields.items():
            values = [sample[name] for sample in samples]
            try:
                batch[name] = collator(values, schema.get(name) if schema else None)
            except TypeError:
                batch[name] = collator(values)
        return batch


class Seq2SeqCollator(object):
    """One batched tokenizer call for source/target text with loss-safe labels."""

    accepts_schema = True

    def __init__(self, tokenizer, source_field="text", target_field="target_text",
                 keep_text=False, label_pad_token_id=-100, tokenizer_kwargs=None,
                 **kwargs):
        if not callable(tokenizer):
            raise TypeError("tokenizer must be callable")
        self.tokenizer = tokenizer
        self.source_field = source_field
        self.target_field = target_field
        self.keep_text = bool(keep_text)
        self.label_pad_token_id = int(label_pad_token_id)
        self.tokenizer_kwargs = dict(tokenizer_kwargs or {})
        self.tokenizer_kwargs.update(kwargs)
        self.tokenizer_kwargs.setdefault("padding", "longest")
        self.tokenizer_kwargs.setdefault("return_tensors", None)

    def __call__(self, samples, schema=None):
        sources = [sample[self.source_field] for sample in samples]
        targets = [sample[self.target_field] for sample in samples]
        encoded = self.tokenizer(sources, text_target=targets,
                                 **self.tokenizer_kwargs)
        encoded = _normalize_tokenizer_output(encoded)
        if "labels" not in encoded:
            raise ValueError("seq2seq tokenizer must return labels for text_target")
        labels = encoded["labels"]
        if isinstance(labels, list):
            labels, _ = _pad_arrays(labels, pad_value=self.label_pad_token_id,
                                     pad_to_multiple_of=self.tokenizer_kwargs.get(
                                         "pad_to_multiple_of"))
        else:
            labels = np.asarray(labels).copy() if isinstance(labels, np.ndarray) else labels
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if isinstance(labels, np.ndarray) and pad_token_id is not None:
            labels[labels == pad_token_id] = self.label_pad_token_id
        encoded["labels"] = labels
        remaining = []
        for sample in samples:
            values = dict(sample)
            if not self.keep_text:
                values.pop(self.source_field, None)
                values.pop(self.target_field, None)
            remaining.append(values)
        batch = default_collate(remaining, schema=schema)
        batch.update(encoded)
        return batch