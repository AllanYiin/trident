import gc
import numbers
import random
import sys
import copy
from collections import namedtuple
from numbers import Number
from typing import Optional, Union, Any, Tuple
import builtins
import numpy as np

from trident.backend.common import get_backend, OrderedDict,to_list,get_device

__all__ = ['ReplayBuffer', 'Rollout', 'ActionStrategy','ObservationType']
if get_backend() == 'pytorch':
    from trident.backend.pytorch_ops import *
elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_ops import *


class ActionStrategy:
    OnPolicy = 'OnPolicy'
    OffPolicy = 'OffPolicy'
    Offline = 'Offline'


class ObservationType:
    Box = 'Box'
    Image = 'Image'
    Tuple = 'Tuple'


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):
    """ A buffer to hold previously-generated states. """

    def __init__(self, capacity, **kwargs):
        """

        Args:
            capacity ()int: Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        args = [tuple([to_numpy(a) for a in arg]) if isinstance(arg, tuple) else to_numpy(arg) for arg in args]

        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.position] = Transition(*args)
        self.position = int(0.5 * self.capacity + (self.position + 1) % (0.5 * self.capacity))

    def sample(self, batch_size):
        """Query the memory to construct batch"""

        transitions = random_choice(self.memory, batch_size)  # list of named tuple  [(x1,y1),(x2,y2),(x3,y3)]

        # list of named tuple to tuple of list [(x1,y1),(x2,y2),(x3,y3)]==>([x1,x2,x3],[y1,y2,y3])
        items = list(zip(*transitions))

        return Transition(*items)

    def __len__(self):
        return len(self.memory)




class SegmentTree:
    """Implementation of Segment Tree.

    The segment tree stores an array ``arr`` with size ``n``. It supports value
    update and fast query of the sum for the interval ``(left, right)`` in
    O(log n) time. The detailed procedure is as follows:

    1. Pad the array to have length of power of 2, so that leaf nodes in the \
    segment tree have the same depth.
    2. Store the segment tree in a binary heap.

    param int size: the size of segment tree.
    """

    def __init__(self, size: int) -> None:
        bound = 1
        while bound < size:
            bound *= 2
        self._size = size
        self._bound = bound
        self._value = np.zeros([bound * 2])
        self._compile()

    def __len__(self) -> int:
        return self._size

    def __getitem__(
            self, index: Union[int, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Return self[index]."""
        return self._value[index + self._bound]

    def __setitem__(
            self, index: Union[int, np.ndarray], value: Union[float, np.ndarray]
    ) -> None:
        """Update values in segment tree.

        Duplicate values in ``index`` are handled by numpy: later index
        overwrites previous ones.

        Examples
        >>> a = np.array([1, 2, 3, 4])
        >>> a[[0, 1, 0, 1]] = [4, 5, 6, 7]
        >>> print(a)
        [6 7 3 4]
        """
        if isinstance(index, int):
            index, value = np.array([index]), np.array([value])
        assert np.all(0 <= index) and np.all(index < self._size)
        _setitem(self._value, index + self._bound, value)

    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        """Return operation(value[start:end])."""
        if start == 0 and end is None:
            return self._value[1]
        if end is None:
            end = self._size
        if end < 0:
            end += self._size
        return _reduce(self._value, start + self._bound - 1, end + self._bound)

    def get_prefix_sum_idx(self, value: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        r"""Find the index with given value.

        Return the minimum index for each ``v`` in ``value`` so that
        :math:`v \le \mathrm{sums}_i`, where
        :math:`\mathrm{sums}_i = \sum_{j = 0}^{i} \mathrm{arr}_j`.

        warning::

            Please make sure all of these values inside the segment tree are
            non-negative when using this function.
        """
        assert np.all(value >= 0.0) and np.all(value < self._value[1])
        single = False
        if not isinstance(value, np.ndarray):
            value = np.array([value])
            single = True
        index = _get_prefix_sum_idx(value, self._bound, self._value)
        return index.item() if single else index

def _get_prefix_sum_idx(
        value: np.ndarray, bound: int, sums: np.ndarray
) -> np.ndarray:
    """Numba version (v0.51), 5x speed up with size=100000 and bsz=64.

    vectorized np: 0.0923 (numpy best) -> 0.024 (now)
    for-loop: 0.2914 -> 0.019 (but not so stable)
    """
    index = np.ones(value.shape, dtype=np.int64)
    while index[0] < bound:
        index *= 2
        lsons = sums[index]
        direct = lsons < value
        value -= lsons * direct
        index += direct
    index -= bound
    return index


def _compile(self) -> None:
    f64 = np.array([0, 1], dtype=np.float64)
    f32 = np.array([0, 1], dtype=np.float32)
    i64 = np.array([0, 1], dtype=np.int64)
    _setitem(f64, i64, f64)
    _setitem(f64, i64, f32)
    _reduce(f64, 0, 1)
    _get_prefix_sum_idx(f64, 1, f64)
    _get_prefix_sum_idx(f32, 1, f64)

def _setitem(tree: np.ndarray, index: np.ndarray, value: np.ndarray) -> None:
    """Numba version, 4x faster: 0.1 -> 0.024."""
    tree[index] = value
    while index[0] > 1:
        index //= 2
        tree[index] = tree[index * 2] + tree[index * 2 + 1]

def _reduce(tree: np.ndarray, start: int, end: int) -> float:
    """Numba version, 2x faster: 0.009 -> 0.005."""
    # nodes in (start, end) should be aggregated
    result = 0.0
    while end - start > 1:  # (start, end) interval is not empty
        if start % 2 == 0:
            result += tree[start + 1]
        start //= 2
        if end % 2 == 1:
            result += tree[end - 1]
        end //= 2
    return result



class Rollout(OrderedDict):
    def __init__(self,enable=True,only_cpu=True, name='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name=name
        self.enable = False
        self.only_cpu=only_cpu
        self.enable = enable

    def __getattr__(self, name):
        if name in super().key_list:
            return super().__getitem__(name)
        elif name in self.__dict__:
            return object.__getattr__(self, name)
        else:
            return None


    def __setattr__(self, name: str, value) -> None:
        if name in ['only_cpu','name','enable'] or name in self.__dict__:
            object.__setattr__(self, name, value)
        elif name in super().key_list:
            super().__setitem__(self,name,value)
        else:
            object.__setattr__ (self,name, value)



    def regist(self,data_name:str):
        if self.enable and data_name not in self.key_list:
            self[data_name]=[]

    def collect(self, data_name: str, value: (float,np.ndarray, Tensor)):
        if self.enable:
            if data_name not in self:
                self.regist(data_name)
            if self.only_cpu:
                if isinstance(value,(numbers.Number,bool,np.bool_)) or value is None:
                    pass
                elif is_tensor(value) and ndim(value)==0:
                    value=value.item()
                elif is_tensor(value):
                    value=value.cpu()
                # else:
                #     value = to_numpy(value)
            self[data_name].append(value)

    def reset(self):
        if self.enable:
            for k in self.key_list:
                del self[k]
                self[k]=[]
            gc.collect()

    def housekeeping(self,num_samples=1000):
        if self.enable:
            for k in self.key_list:
                if len(self[k])>num_samples:
                    self[k]=[ self[k][n]  for n in range(len(self[k])) if n>=len(self[k])-num_samples]
            gc.collect()


    def get(self,data_name):
        if self.enable and data_name in self and len(self[data_name]) > 0:
            return [self[data_name][idx].copy().to(get_device()) if is_tensor(self[data_name][idx]) else copy.deepcopy(v[idx]) for idx in range(len(self[data_name]))]
        else:
            return []

    def get_last(self,data_name):
        if self.enable and data_name in self and len(self[data_name])>0:
            return self[data_name][-1]
        else:
            return None

    def get_reverse(self,data_name):
        if self.enable and data_name in self and len(self[data_name])>0:
            return [self[data_name][idx].copy().to(get_device()) if is_tensor(self[data_name][idx]) else copy.deepcopy(v[idx]) for idx in range(len(self[data_name]))][::-1]
        else:
            return []

    def get_normalized(self,data_name):

        if self.enable and data_name in self and len(self[data_name])>3:
            try:
                if is_tensor(self[data_name][0]):
                    merge=concate(self[data_name],axis=0)
                    merge_mean,merge_std=moments(merge,axis=0)
                    merge=(merge-merge_mean)/merge_std
                    return to_list(merge)
                elif isinstance(self[data_name][0],np.ndarray):
                    merge=np.concatenate(self[data_name],axis=0)
                    merge=(merge-merge.mean())/merge.std()
                    return to_list(merge)
                elif isinstance(self[data_name][0],numbers.Number):
                    merge = np.array(self[data_name])
                    merge=(merge-merge.mean())/merge.std()
                    return to_list(merge)
            except:
                return self[data_name]
        elif self.enable:
            return self[data_name]
        else:
            return []

    def get_running_mean(self, data_name):

        if self.enable and data_name in self and len(self[data_name]) > 3:
            try:
                if isinstance(self[data_name][0], numbers.Number):
                    total_mean=builtins.sum(self[data_name])/float(builtins.max(len(self[data_name]),1))
                    result=[]
                    running_sum=0
                    for i in range(len(self[data_name])):
                        running_sum+=self[data_name][i]
                        result.append((running_sum/(i+1))-total_mean)
                 
                    return result
                else:
                    raise ValueError('Only numeric data is supporte')
            except:
                return self[data_name]
        elif self.enable:
            return self[data_name]
        else:
            return []

    def get_samples(self, batch_size=8, recall_last=True):
        if self.enable:
            batch_size=builtins.min(batch_size,len(self.value_list[0]))
            can_sample=len(list(set([len(v) for v in self.value_list])))==1
            if recall_last  :
                indexs = random.choices(list(range(len(self.value_list[0]))),k=batch_size if not recall_last else int(0.75 * batch_size))
                indexs.extend(list(range(len(self.value_list[0])))[-1*(batch_size//4):])
            else:
                indexs = random.choices(list(range(len(self.value_list[0]))),k=batch_size )
            return_data=OrderedDict()
            if can_sample:
                for k,v in self.items():
                    return_data[k]=[v[idx].copy().to(get_device()) if is_tensor(v[idx]) else copy.deepcopy(v[idx]) for idx in indexs]
                return return_data
            else:
                raise ValueError('get_sample need all collection have same length.')
        else:
            return OrderedDict()

    def get_snapshot(self):
        new_rollout=Rollout()
        for k,v in self.items():
            new_rollout[k]=copy.deepcopy(v)
        return new_rollout

    def __len__(self):
        if len(self.value_list)==0:
            return 0
        return builtins.max([len(v) for v in self.value_list])








# class PrioritizedReplayBuffer(ReplayBuffer):
#     """Implementation of Prioritized Experience Replay. arXiv:1511.05952.
#
#     :param float alpha: the prioritization exponent.
#     :param float beta: the importance sample soft coefficient.
#
#     .. seealso::
#
#         Please refer to :class:`~tianshou.data.ReplayBuffer` for more detailed
#         explanation.
#     """
#
#     def __init__(
#         self, capacity: int, alpha: float, beta: float, **kwargs: Any
#     ) -> None:
#         super().__init__(capacity, **kwargs)
#         assert alpha > 0.0 and beta >= 0.0
#         self._alpha, self._beta = alpha, beta
#         self._max_prio = self._min_prio = 1.0
#         # save weight directly in this class instead of self._meta
#         self.weight = SegmentTree(capacity)
#         self.__eps = np.finfo(np.float32).eps.item()
#
#     def push(
#         self,
#         obs: Any,
#         act: Any,
#         rew: Union[Number, np.number, np.ndarray],
#         done: Union[Number, np.number, np.bool_],
#         obs_next: Any = None,
#         info= {},
#         policy= {},
#         weight: Optional[Union[Number, np.number]] = None,
#         **kwargs: Any,
#     ) -> None:
#         """Add a batch of data into replay buffer."""
#         if weight is None:
#             weight = self._max_prio
#         else:
#             weight = np.abs(weight)
#             self._max_prio = max(self._max_prio, weight)
#             self._min_prio = min(self._min_prio, weight)
#         self.weight[self._index] = weight ** self._alpha
#         super().add(obs, act, rew, done, obs_next, info, policy, **kwargs)
#
#     def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
#         """Get a random sample from buffer with priority probability.
#
#         Return all the data in the buffer if batch_size is 0.
#
#         :return: Sample data and its corresponding index inside the buffer.
#
#         The "weight" in the returned Batch is the weight on loss function
#         to de-bias the sampling process (some transition tuples are sampled
#         more often so their losses are weighted less).
#         """
#         assert self._size > 0, "Cannot sample a buffer with 0 size!"
#         if batch_size == 0:
#             indice = np.concatenate([
#                 np.arange(self._index, self._size),
#                 np.arange(0, self._index),
#             ])
#         else:
#             scalar = np.random.rand(batch_size) * self.weight.reduce()
#             indice = self.weight.get_prefix_sum_idx(scalar)
#         batch = self[indice]
#         # important sampling weight calculation
#         # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
#         # simplified formula: (p_j/p_min)**(-beta)
#         batch.weight = (batch.weight / self._min_prio) ** (-self._beta)
#         return batch, indice
#
#     def update_weight(
#         self,
#         indice: Union[np.ndarray],
#         new_weight: Union[np.ndarray, torch.Tensor]
#     ) -> None:
#         """Update priority weight by indice in this buffer.
#
#         :param np.ndarray indice: indice you want to update weight.
#         :param np.ndarray new_weight: new priority weight you want to update.
#         """
#         weight = np.abs(to_numpy(new_weight)) + self.__eps
#         self.weight[indice] = weight ** self._alpha
#         self._max_prio = max(self._max_prio, weight.max())
#         self._min_prio = min(self._min_prio, weight.min())
#
#     def __getitem__(
#         self, index: Union[slice, int, np.integer, np.ndarray]
#     ) -> Batch:
#         return Batch(
#             obs=self.get(index, "obs"),
#             act=self.act[index],
#             rew=self.rew[index],
#             done=self.done[index],
#             obs_next=self.get(index, "obs_next"),
#             info=self.get(index, "info"),
#             policy=self.get(index, "policy"),
#             weight=self.weight[index],
#         )
