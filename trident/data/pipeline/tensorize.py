from __future__ import absolute_import, division, print_function

import numpy as np

try:
    from collections.abc import Mapping
except ImportError:  # pragma: no cover - Python 3.5
    from collections import Mapping

from .batch import Batch
from .collate import RaggedBatch


def _convert(value, converter):
    if isinstance(value, RaggedBatch):
        return RaggedBatch(_convert(value.values, converter),
                           _convert(value.row_splits, converter))
    if isinstance(value, Mapping):
        converted = dict((key, _convert(item, converter))
                         for key, item in value.items())
        return Batch(converted, schema=getattr(value, "schema", None),
                     metadata=getattr(value, "metadata", None)) if isinstance(value, Batch) else converted
    if isinstance(value, tuple):
        return tuple(_convert(item, converter) for item in value)
    if isinstance(value, list):
        return [_convert(item, converter) for item in value]
    return converter(value)


class TorchTensorizer(object):
    def __init__(self, copy=False):
        self.copy = bool(copy)

    def __call__(self, batch):
        try:
            import torch
        except ImportError:
            raise ImportError("TorchTensorizer requires PyTorch")

        def converter(value):
            if isinstance(value, np.ndarray) and value.dtype.kind not in ("O", "U", "S"):
                return torch.tensor(value) if self.copy else torch.from_numpy(value)
            if isinstance(value, np.generic):
                return torch.as_tensor(value)
            return value
        return _convert(batch, converter)


class TensorFlowTensorizer(object):
    def __call__(self, batch):
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlowTensorizer requires TensorFlow")

        def converter(value):
            if isinstance(value, np.ndarray) and value.dtype.kind != "O":
                return tf.convert_to_tensor(value)
            if isinstance(value, np.generic):
                return tf.convert_to_tensor(value)
            return value
        return _convert(batch, converter)
