from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import locale
import os
import random
import numpy as np
import warnings
import itertools

from ..backend.common import OrderedDict


__all__ = ['Sampler','SequentialSampler','RandomSampler','BatchSampler']

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

    Arguments:
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

    Arguments:
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
            return iter(np.random.randint(high=n, low=0,size=(self.bootstrap_samples), dtype=np.int64).tolist())
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

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, data_source, batch_size=1, is_shuffle=True,drop_last=False):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.is_shuffle=is_shuffle
        self.image_transforms=[]
        self.label_transforms = []

        idxes = np.arange(len(self.data_source))
        if len(self.data_source) % self.batch_size>0:
            idxes=idxes[:-(len(self.data_source) % self.batch_size)]
        if self.is_shuffle==True:
            np.random.shuffle(idxes)
        idxes = list(idxes)

        self.sampler = itertools.cycle(iter(idxes))

    def __iter__(self):
        batch =OrderedDict()
        _data_cnt=0
        for idx in self.sampler:
            try:
                _return_data=self.data_source[idx]
                if _return_data[0] is not None:
                    for i in range(len(_return_data)):
                        if i not in batch:
                            batch[i]=[]
                        batch[i].append(_return_data[i])
                _data_cnt+=1
            except Exception as e:
                print(e)

            if _data_cnt== self.batch_size:
                yield tuple([np.array(v) for k,v in batch.items()])
                batch = {}
                _data_cnt = 0
        if len(batch)==0:
            raise StopIteration

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source)+ self.batch_size - 1) // self.batch_size

    def reset(self):
        idxes = np.arange(len(self.data_source))
        if len(self.data_source) % self.batch_size > 0:
            idxes = idxes[:-(len(self.data_source) % self.batch_size)]
        if self.is_shuffle == True:
            np.random.shuffle(idxes)
        idxes = list(idxes)
        self.sampler = iter(idxes)

