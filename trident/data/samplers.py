from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import builtins
import copy
import inspect
import itertools
import locale
import os
import random
import warnings
import numbers
import numpy as np
from trident.backend.tensorspec import TensorSpec,ObjectType

from trident.data.image_common import check_same_size
from trident.backend.common import OrderedDict,get_backend

__all__ = ['Sampler', 'SequentialSampler', 'RandomSampler', 'BatchSampler']


class Sampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        super(SequentialSampler, self).__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, is_bootstrap=False, bootstrap_samples=None):
        super(RandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.is_bootstrap = is_bootstrap
        self.bootstrap_samples = bootstrap_samples

        if self.bootstrap_samples is not None and is_bootstrap is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.bootstrap_samples is None:
            self.bootstrap_samples = len(self.data_source)

        if not isinstance(self.bootstrap_samples, int) or self.bootstrap_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.bootstrap_samples))
        if not isinstance(self.is_bootstrap, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.is_bootstrap))

    def __iter__(self):
        n = len(self.data_source)
        if self.is_bootstrap:
            return iter(np.random.randint(high=n, low=0, size=(self.bootstrap_samples), dtype=np.int64).tolist())
        return iter(np.random.randperm(n).tolist())

    def __len__(self):
        return len(self.data_source)


class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Examples:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, data_source, batch_size=1, is_shuffle=True, drop_last=True, sample_filter=None,dynamic_padding=False):
        super().__init__(data_source)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.is_shuffle = is_shuffle
        self.dynamic_padding=dynamic_padding



        self.reset()
        self.sample_filter = None
        if inspect.isfunction(sample_filter) or callable(sample_filter):
            self.sample_filter = sample_filter



    def _collate(self, rows):
        batch = copy.deepcopy(self.data_source.data_template)
        sequence_fields = []
        for spec, values in zip(batch, zip(*rows)):
            if not check_same_size(*values):
                raise ValueError("incompatible sample shapes for field {0}".format(spec.name))
            batch[spec] = np.asarray(values)
            parent = self.data_source.parent
            if (parent is not None and parent.dynamic_padding and
                    batch[spec].ndim >= 2 and batch[spec].dtype.kind in "iu"):
                sequence_fields.append(spec)
        if sequence_fields:
            last = 0
            for spec in sequence_fields:
                positions = np.nonzero(batch[spec])[1]
                if positions.size:
                    last = builtins.max(last, int(positions.max()))
            for spec in sequence_fields:
                batch[spec] = batch[spec][:, :last + 1]
        return tuple(batch.values()) if self.data_source.mode == 'tuple' else batch

    def _batches(self, indices):
        rows = []
        for index in indices:
            values = self.data_source[index].value_list
            if self.sample_filter is not None and not self.sample_filter(values):
                continue
            rows.append(values)
            if len(rows) == self.batch_size:
                yield self._collate(rows)
                rows = []
        if rows and not self.drop_last:
            yield self._collate(rows)

    def _epoch_indices(self):
        indices = np.arange(len(self.data_source))
        if self.is_shuffle:
            np.random.shuffle(indices)
        return indices

    def iter_epoch(self):
        """One finite pass for training, including a short last batch."""
        return self._batches(self._epoch_indices())

    def __iter__(self):
        # The historical next() API fills batches across epoch boundaries.
        return self._batches(self.sampler)

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def reset(self):
        self.sampler = itertools.cycle(self._epoch_indices())
