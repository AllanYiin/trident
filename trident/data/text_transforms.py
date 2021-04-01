import builtins
import copy
import math
import numbers
import random
import inspect
import string
from functools import wraps
from typing import Sequence, Tuple, Dict, Union, Optional
import collections
import  numpy as np

from trident.backend.tensorspec import TensorSpec, object_type_inference, ObjectType


from trident.backend.common import OrderedDict
from trident.backend.common import *
from trident.backend.tensorspec import TensorSpec,object_type_inference

if get_backend() == 'pytorch':
    from trident.backend.pytorch_ops import *
elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_ops import *
from trident.data.transform import TextTransform

__all__ = ['RandomSwapChar','RandomInsertChar','ToHalfWidth']





class RandomSwapChar(TextTransform):
    """
    Swap two characters in given string randomly.

    Args:
        probs (float): probability to swap characters
        name (string): transformation name.

    Examples:
        >>> swapchar=RandomSwapChar(probs=0.1)
        >>> data=list("我給妳的愛寫在西元前深埋在美索不達米亞平原")
        >>> result=swapchar(data)
        >>> print(result)
        >>> data1=np.arange(0,30)
        >>> result1=swapchar(data1)
        >>> print(result1)
    """

    def __init__(self, probs=0.05,name='random_swap_char',**kwargs):
        super().__init__()
        self.probs = probs
        self.name=name

    def _apply_corpus(self, corpus,spec:TensorSpec):
        self._idxes=self._get_shape(corpus)
        if isinstance(corpus,(str,list)):
            return [corpus[i] for i in self._idxes]
        elif isinstance(corpus,np.ndarray):
            idx=np.array(self._idxes).astype(np.int64)
            return corpus[idx]

    def _apply_sequence_labels(self, labels,spec:TensorSpec):
        if isinstance(labels,(str,list)):
            return [labels[i] for i in self._idxes]
        elif isinstance(labels,np.ndarray):
            idx=np.array(self._idxes).astype(np.int64)
            return labels[idx]

    def _apply_sequence_mask(self, mask,spec:TensorSpec):
        if isinstance(mask, (str, list)):
            return [mask[i] for i in self._idxes]
        elif isinstance(mask, np.ndarray):
            idx = np.array(self._idxes).astype(np.int64)
            return mask[idx]

    def _get_shape(self, corpus):
        idxes=list(range(len(corpus)))
        last_swap_idx=-1
        for i in range(len(corpus)):
            if random.random()<self.probs and i+1<len(corpus) and i>last_swap_idx+1:
                swap_block=copy.deepcopy(idxes[i:i+2])
                swap_block.reverse()
                idxes[i]= swap_block[0]
                idxes[i+1] = swap_block[1]
                last_swap_idx=i+1
        return idxes

class RandomInsertChar(TextTransform):
    """
    Swap two characters in given string randomly.

    Args:
        probs (float): probability to swap characters
        name (string): transformation name.

    Examples:
        >>> insertchar=RandomInsertChar(probs=0.1,chars='的了若可然也或 ,.')
        >>> data=list("我給妳的愛寫在西元前深埋在美索不達米亞平原")
        >>> result=insertchar(data)
        >>> print(result)

    """

    def __init__(self, probs=0.05,chars=' ' ,name='random_insert_char',**kwargs):
        super().__init__()
        self.probs = probs
        self.chars=list(chars)
        self.name=name

    def _apply_corpus(self, corpus,spec:TensorSpec):
        self._idxes=self._get_shape(corpus)
        if isinstance(corpus,(str,list)):
            for i in range(len(self._idxes)):
                pos=self._idxes[i]+i
                corpus.insert(pos,random_choice(self.chars,1))
            return corpus
        elif isinstance(corpus,np.ndarray):
            return corpus

    def _apply_sequence_labels(self, labels,spec:TensorSpec):
        if isinstance(labels,(str,list)):
            for i in range(len(self._idxes)):
                pos = self._idxes[i] + i
                labels.insert(pos, labels[pos-1])
            return labels
        elif isinstance(labels,np.ndarray):
            idx=np.array(self._idxes).astype(np.int64)
            return labels

    def _apply_sequence_mask(self, mask,spec:TensorSpec):
        if isinstance(mask, (str, list)):
            for i in range(len(self._idxes)):
                pos = self._idxes[i] + i
                mask.insert(pos, mask[pos - 1])
            return mask
        elif isinstance(mask, np.ndarray):
            idx = np.array(self._idxes).astype(np.int64)
            return mask

    def _get_shape(self, corpus):
        insert_pos = []
        if isinstance(corpus,(str,list)):
            idxes=list(range(len(corpus)))
            last_insert_idx=-1
            for i in range(len(corpus)):
                if random.random()<self.probs and i+1<len(corpus) and i>last_insert_idx+1 and i>0:
                    insert_pos.append(i)
                    last_swap_idx=i+1
            return insert_pos
        else:
            return insert_pos


class ToHalfWidth(TextTransform):
    """
    Swap two characters in given string randomly.

    Args:
        probs (float): probability to swap characters
        name (string): transformation name.

    Examples:
        >>> tohalf=ToHalfWidth()
        >>> data=list("一二三四五六七１２３４５６７")
        >>> result=tohalf(data)
        >>> print(result)
        一二三四五六七1234567

    """

    def __init__(self, name='random_swap_char',**kwargs):
        super().__init__()

        self.name=name

    def _apply_corpus(self, corpus,spec:TensorSpec):
        out_str = []
        for char in corpus:
            inside_code = ord(char)
            if inside_code == 0x3000 or inside_code == 12288 or char == string.whitespace:  # 全形空格直接轉換
                out_str.append(' ')
            elif 65281 <= inside_code <= 65374:
                inside_code -= 0xfee0
                out_str.append(chr(inside_code))
            else:
                out_str.append(char)

        return ''.join(out_str)

    def _apply_sequence_labels(self, labels,spec:TensorSpec):
       return labels

    def _apply_sequence_mask(self, mask,spec:TensorSpec):
       return mask

