from __future__ import absolute_import, division, print_function

from .batch import Batch
from .collate import RaggedBatch


def to_torch_dataloader(iterator, **kwargs):
    """Expose a configured pipeline Iterator through PyTorch DataLoader.

    Batching and worker execution remain owned by the pipeline, so the outer
    DataLoader deliberately uses ``batch_size=None`` and ``num_workers=0``.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("to_torch_dataloader requires PyTorch")
    if kwargs.pop("batch_size", None) is not None:
        raise ValueError("batch_size belongs to pipeline Iterator")
    if int(kwargs.pop("num_workers", 0)) != 0:
        raise ValueError("workers belong to pipeline Iterator")

    class _PipelineIterableDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(iterator)

        def __len__(self):
            return len(iterator)

    return torch.utils.data.DataLoader(
        _PipelineIterableDataset(), batch_size=None, num_workers=0, **kwargs)


def _tf_value(value, tf):
    if isinstance(value, RaggedBatch):
        return tf.RaggedTensor.from_row_splits(
            _tf_value(value.values, tf), _tf_value(value.row_splits, tf))
    if isinstance(value, dict):
        return dict((key, _tf_value(item, tf)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_tf_value(item, tf) for item in value)
    return tf.convert_to_tensor(value)


def _relax_batch_dimension(spec, tf):
    if isinstance(spec, tf.TensorSpec) and spec.shape.rank:
        return tf.TensorSpec([None] + spec.shape.as_list()[1:], spec.dtype,
                             name=spec.name)
    if isinstance(spec, tf.RaggedTensorSpec) and spec.shape.rank:
        return tf.RaggedTensorSpec(
            [None] + spec.shape.as_list()[1:], spec.dtype,
            ragged_rank=spec.ragged_rank, row_splits_dtype=spec.row_splits_dtype)
    return spec


def to_tensorflow_dataset(iterator, output_signature=None, prefetch=1):
    """Create a ``tf.data.Dataset`` while keeping pipeline epoch semantics."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("to_tensorflow_dataset requires TensorFlow")

    if output_signature is None:
        state = iterator.state_dict()
        probe_iterator = iter(iterator)
        try:
            probe = next(probe_iterator)
        except StopIteration:
            raise ValueError("cannot infer output_signature from an empty Iterator")
        finally:
            close = getattr(probe_iterator, "close", None)
            if callable(close):
                close()
        iterator.load_state_dict(state)
        inferred = tf.nest.map_structure(tf.type_spec_from_value,
                                         _tf_value(dict(probe), tf))
        output_signature = tf.nest.map_structure(
            lambda spec: _relax_batch_dimension(spec, tf), inferred)

    def generator():
        for batch in iterator:
            values = dict(batch) if isinstance(batch, Batch) else batch
            yield _tf_value(values, tf)

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=output_signature)
    if prefetch:
        dataset = dataset.prefetch(int(prefetch))
    return dataset
