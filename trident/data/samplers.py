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
from trident.backend.tensorspec import TensorSpec

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



        idxes = np.arange(len(self.data_source.data))
        if len(self.data_source) % self.batch_size > 0:
            idxes = idxes[:-(len(self.data_source.data) % self.batch_size)]
        if self.is_shuffle:
            n = len(self.data_source)
            random_range=np.arange(n)
            np.random.shuffle(random_range)
            self.sampler = itertools.cycle(iter(random_range.tolist()))

        else:
            self.sampler =itertools.cycle(iter(range(len(self.data_source))))
        self.sample_filter = None
        if inspect.isfunction(sample_filter) or callable(sample_filter):
            self.sample_filter = sample_filter



    def __iter__(self):

        batch_data = []
        for idx in self.sampler:
            try:

                _return_data = self.data_source[idx]
                # filter sample
                if self.sample_filter is None or self.sample_filter(_return_data.value_list):
                    batch_data.append(_return_data.value_list)

                    if len(batch_data) == self.batch_size:
                        returnData = copy.deepcopy(self.data_source.data_template)
                        unzip_batch_data = list(zip(*batch_data))

                        _dynamaic_paddings=[]
                        for i in range(len(unzip_batch_data)):
                            if check_same_size(*unzip_batch_data[i]):
                                try:
                                    if all([isinstance(s,str) for s in unzip_batch_data[i]]):
                                        returnData[returnData.key_list[i]] = np.array([array for array in unzip_batch_data[i]],dtype=np.string_)
                                    else:
                                        returnData[returnData.key_list[i]] = np.array([array for array in unzip_batch_data[i] ])
                                        if  self.data_source.parent is not None and self.data_source.parent .dynamic_padding and 'int' in  str(returnData[returnData.key_list[i]] .dtype):
                                             _dynamaic_paddings.append(np.max(np.argwhere(returnData[returnData.key_list[i]]!=0)[:,1]))
                                except Exception as e:
                                    print([array.shape for array in unzip_batch_data[i] ])
                            else:
                                print([array.shape for array in unzip_batch_data[i] ])
                                batch_data=[]
                        if self.data_source.parent is not None and self.data_source.parent .dynamic_padding:
                            final_sequence_length=builtins.max(_dynamaic_paddings)
                            for k in returnData.key_list:
                                if 'int' in  str(returnData[k] .dtype):
                                    returnData[k]=returnData[k][:,:final_sequence_length+1]


                        if self.data_source.mode=='tuple':
                            yield tuple(returnData.value_list)
                        elif self.data_source.mode=='dict':
                            yield returnData
                        batch_data = []
            except Exception as e:
                print('index:{0} get fail.'.format(idx))
                print(e)

        if len(batch_data) > 0 and not self.drop_last:
            returnData = copy.deepcopy(self.data_source.data_template)
            unzip_batch_data = list(zip(*batch_data))
            for i in range(len(unzip_batch_data)):
                if check_same_size(*unzip_batch_data[i]):
                    try:
                        if all([isinstance(s, str) for s in unzip_batch_data[i]]):
                            returnData[returnData.key_list[i]] = np.array([array for array in unzip_batch_data[i]], dtype=np.string_)
                        else:
                            returnData[returnData.key_list[i]] = np.array([array for array in unzip_batch_data[i]])
                    except Exception as e:
                        print([array.shape for array in unzip_batch_data[i]])
                else:
                    print([array.shape for array in unzip_batch_data[i]])
                    batch_data = []
            if self.data_source.mode == 'tuple':
                yield tuple(returnData.value_list)
            elif self.data_source.mode == 'dict':
                yield returnData

            batch_data = []
        self.reset()
        # raise StopIteration

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def reset(self):
        idxes = np.arange(len(self.data_source))
        if len(self.data_source) % self.batch_size > 0:
            idxes = idxes[:-(len(self.data_source) % self.batch_size)]
        if self.is_shuffle:
            np.random.shuffle(idxes)
        idxes = list(idxes)
        self.sampler = iter(idxes)


