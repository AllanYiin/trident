import builtins
import copy
import numbers
import os
import random
import string

import numpy as np
from trident.backend import numpy_ops
from trident.backend.common import *
from trident.backend.tensorspec import TensorSpec
from trident.data.transform import TextTransform
from trident.data.utils import download_file_from_google_drive, unpickle


if get_backend() == 'pytorch':
    #from trident.backend.pytorch_backend import get_device
    from trident.backend.pytorch_ops import *
elif get_backend() == 'tensorflow':
    #from trident.backend.tensorflow_backend import get_device
    from trident.backend.tensorflow_ops import *



__all__ = ['bpmf_phonetic', 'numbers_string', 'alphabets', 'chinese_characters', 'RandomSwapChar', 'RandomInsertChar', 'ToHalfWidth', 'RandomMask', 'BopomofoConvert', 'ChineseConvert',
           'RandomHomophonicTypo', 'RandomHomomorphicTypo', 'VocabsMapping']

bpmf_phonetic = 'ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩ'
numbers_string= '0123456789'
alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# Ideographic number zero+CJK Unified Ideographs
chinese_characters = '\u3007'.encode('utf-8').decode('utf-8') + ''.join([chr(i) for i in range(19968, 40959)])


# CJK Unified Ideographs Extension A
# ch2=''.join([chr(i) for i in range(13312,19903)] )
# CJK Compatibility Ideographs
# ch3=''.join([chr(i) for i in range(63744,64255)] )

# if sys.maxunicode > 0xFFFF:
#     chinese_characters += (
#         '\U00020000-\U0002A6DF'  # CJK Unified Ideographs Extension B
#         '\U0002A700-\U0002B73F'  # CJK Unified Ideographs Extension C
#         '\U0002B740-\U0002B81F'  # CJK Unified Ideographs Extension D
#         '\U0002F800-\U0002FA1F'  # CJK Compatibility Ideographs Supplement
#     )
#
# chinese_radicals = (
#     '\u2F00-\u2FD5'  # Kangxi Radicals
#     '\u2E80-\u2EF3'  # CJK Radicals Supplement
# )


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

    def __init__(self, probs=0.05, name='random_swap_char', skip_first=False, skip_last=False, **kwargs):
        super().__init__()
        self.probs = probs
        self.name = name
        self.skip_first = skip_first
        self.skip_last = skip_last

    def _apply_corpus(self, corpus, spec: TensorSpec):
        self._idxes = self._get_shape(corpus)
        if isinstance(corpus, (str, list)):
            return ''.join([corpus[i] for i in self._idxes])
        elif isinstance(corpus, np.ndarray):
            idx = np.array(self._idxes).astype(np.int64)
            return corpus[idx]

    def check_swaptable(self, corpus, idx):
        def check_single(corpus, idx):
            char = corpus[idx]
            if char not in string.digits and char not in string.punctuation and char not in string.ascii_letters and char not in bpmf_phonetic:
                return True
            else:
                return False

        return all([check_single(corpus, i) for i in range(idx - 1, idx + 3) if i < len(corpus)])

    def _apply_sequence_labels(self, labels, spec: TensorSpec):
        if isinstance(labels, (str, list)):
            return ''.join([labels[i] for i in self._idxes])
        elif isinstance(labels, np.ndarray):
            idx = np.array(self._idxes).astype(np.int64)
            return labels[idx]

    def _apply_sequence_mask(self, mask, spec: TensorSpec):
        if isinstance(mask, (str, list)):
            return ''.join([mask[i] for i in self._idxes])
        elif isinstance(mask, np.ndarray):
            idx = np.array(self._idxes).astype(np.int64)
            return mask[idx]

    def _get_shape(self, corpus):
        idxes = list(range(len(corpus)))
        last_swap_idx = -1
        for i in range(len(corpus)):
            if (i == 0 and not self.skip_first) or (i == len(corpus) - 1 and not self.skip_last or 0 < i < len(corpus) - 1):
                if random.random() < self.probs and i + 1 < len(corpus) and i > last_swap_idx + 1:
                    if isinstance(corpus, np.ndarray) or self.check_swaptable(corpus, i):
                        swap_block = copy.deepcopy(idxes[i:i + 2])
                        swap_block.reverse()
                        idxes[i] = swap_block[0]
                        idxes[i + 1] = swap_block[1]
                        last_swap_idx = i + 1
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

    def __init__(self, probs=0.05, chars=' ', name='random_insert_char', **kwargs):
        super().__init__()
        self.probs = probs
        self.chars = list(chars)
        self.name = name

    def _apply_corpus(self, corpus, spec: TensorSpec):
        self._idxes = self._get_shape(corpus)
        if isinstance(corpus, (str, list)):
            for i in range(len(self._idxes)):
                pos = self._idxes[i] + i
                corpus.insert(pos, random_choice(self.chars, 1))
            return corpus
        elif isinstance(corpus, np.ndarray):
            return corpus

    def _apply_sequence_labels(self, labels, spec: TensorSpec):
        if isinstance(labels, (str, list)):
            for i in range(len(self._idxes)):
                pos = self._idxes[i] + i
                labels.insert(pos, labels[pos - 1])
            return labels
        elif isinstance(labels, np.ndarray):
            idx = np.array(self._idxes).astype(np.int64)
            return labels

    def _apply_sequence_mask(self, mask, spec: TensorSpec):
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
        if isinstance(corpus, (str, list)):
            idxes = list(range(len(corpus)))
            last_insert_idx = -1
            for i in range(len(corpus)):
                if random.random() < self.probs and i + 1 < len(corpus) and i > last_insert_idx + 1 and i > 0:
                    insert_pos.append(i)
                    last_swap_idx = i + 1
            return insert_pos
        else:
            return insert_pos


# RandomDeleteChar
# RandomReplaceChar

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

    def __init__(self, name='random_swap_char', **kwargs):
        super().__init__()
        self.additional_rules = {913: 'A',
                                 914: 'B',
                                 917: 'E',
                                 919: 'H',
                                 921: 'I',
                                 922: 'K',
                                 924: 'M',
                                 925: 'N',
                                 927: 'O',
                                 929: 'P',
                                 932: 'T',
                                 935: 'X',
                                 933: 'Y',
                                 918: 'Z'}
        self.name = name

    def _apply_corpus(self, corpus, spec: TensorSpec):
        out_str = []
        is_list = isinstance(corpus, list)
        for word in corpus:
            if word in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                out_str.append(word)
            else:
                if len(word) > 0:
                    for char in word:
                        try:
                            inside_code = ord(char)
                            if inside_code == 0x3000 or inside_code == 12288 or char == string.whitespace:  # 全形空格直接轉換
                                out_str.append(' ')
                            elif inside_code in self.additional_rules:
                                out_str.append(self.additional_rules[inside_code])
                            elif 65281 <= inside_code <= 65374:
                                inside_code -= 0xfee0
                                out_str.append(chr(inside_code))
                            else:
                                out_str.append(char)
                        except:
                            out_str.append(char)

        return ''.join(out_str) if not is_list else out_str

    def _apply_sequence_labels(self, labels, spec: TensorSpec):
        return labels

    def _apply_sequence_mask(self, mask, spec: TensorSpec):
        return mask


class ChineseConvert(TextTransform):
    """
    Swap two characters in given string randomly.

    Args:
        probs (float): probability to swap characters
        name (string): transformation name.

    Examples:
        >>> cc=ChineseConvert(convert_to='simplified', convert_ratio=0.5)
        >>> data='還是選手自我保護意識不夠'
        >>> print(cc(data))
        还是选手自我保护意识不够
        >>> print(cc._text_info)

    """

    def __init__(self, convert_to='simplified', convert_ratio=1.0, name='chinese_convert', **kwargs):
        super().__init__()
        if convert_to not in ['traditional', 'simplified']:
            raise ValueError('Only traditional, simplified is valid convert_to option.')
        self.convert_to = convert_to
        self.convert_ratio = convert_ratio
        self.traditional2simplified = {}
        self.simplified2traditional = {}
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 't2s.txt'), 'r', encoding='utf-8-sig') as f:
            for line in f.readlines():
                words = line.strip().split('\t')
                if len(words) == 2:
                    self.traditional2simplified[words[0]] = words[1]
                    self.simplified2traditional[words[1]] = words[0]

        self.name = name

    def _apply_corpus(self, corpus, spec: TensorSpec):
        is_list = isinstance(corpus, list)
        update_list=self._text_info.copy()
        out_str = []
        for idx in range(len(corpus)):
            word = corpus[idx]
            if word in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                out_str.append(word)
            elif len(word) > 0:
                for idx2 in range(len(word)):
                    char = word[idx2]
                    if (idx,idx2) in update_list:
                        if self.convert_to == 'simplified' and char in self.traditional2simplified:
                            out_str.append(self.traditional2simplified[char])
                        elif self.convert_to == 'traditional' and char in self.simplified2traditional:
                            out_str.append(self.simplified2traditional[char])
                        else:
                            out_str.append(char)
                        update_list.remove((idx,idx2))

                    else:
                        out_str.append(char)
        return ''.join(out_str) if not is_list else out_str

    def _apply_sequence_labels(self, labels, spec: TensorSpec):
        return labels

    def _apply_sequence_mask(self, mask, spec: TensorSpec):
        return mask

    def _precalculate(self, textdata, **kwargs):

        is_list = isinstance(textdata, list)
        update_list = []
        for idx in range(len(textdata)):
            word=textdata[idx]
            if word in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                pass
            elif len(word) > 0:
                for idx2 in range(len(word)):
                    char =word[idx2]
                    if self.convert_to == 'simplified' and char  in self.traditional2simplified:
                        update_list.append((idx,idx2))
                    elif self.convert_to == 'traditional' and char  in self.simplified2traditional:
                        update_list.append((idx,idx2))
                    elif char in chinese_characters:
                        update_list.append((idx, idx2))
                    #else:
                      #  print(char,flush=True)


        if self.convert_ratio>=1:
            self._text_info=update_list
        elif self.convert_ratio<=0:
            self._text_info =[]
        else:
            num_updates = int(self.convert_ratio * len(update_list))
            self._text_info = numpy_ops.random_choice(update_list,num_updates)






class VocabsMapping(TextTransform):
    """
    Mapping

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

    def __init__(self, mapping_dict=None, output_type='list', unkown_token=None, pad_token=None, cls_token=None, sep_token=None, name='random_swap_char', **kwargs):
        super().__init__()
        if output_type in ['list', 'array', 'string']:
            self.output_type = output_type
        else:
            raise ValueError('Only [list,array] are valid output_type option. ')
        self.mapping_dict = mapping_dict
        self.unkown_token = unkown_token
        if self.unkown_token is None and isinstance(self.mapping_dict, dict) and '[UNK]' in self.mapping_dict:
            self.unkown_token = '[UNK]'
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        if self.pad_token is None and isinstance(self.mapping_dict, dict) and '[PAD]' in self.mapping_dict:
            self.pad_token = '[PAD]'

        self.name = name

    def _apply_corpus(self, corpus, spec: TensorSpec):
        out_str = []
        is_list = isinstance(corpus, list)
        if self.mapping_dict is None or len(self.mapping_dict) == 0:
            return corpus
        else:
            seq_len= len(corpus)
            for k in range(seq_len):
                word=corpus[k] if k< len(corpus) else self.pad_token
                if word in self.mapping_dict:
                    out_str.append(self.mapping_dict[word])
                else:
                    sample_item = list(self.mapping_dict.values())[0]
                    if word in [self.cls_token, self.sep_token, self.pad_token]:
                        if isinstance(sample_item, str):
                            out_str.append(word)
                        elif isinstance(sample_item, np.ndarray):
                            out_str.append(np.zeros_like(sample_item))
                        elif isinstance(sample_item, numbers.Integral):
                            out_str.append(0)
                        elif isinstance(sample_item, numbers.Number):
                            out_str.append(0.0)

                    elif self.unkown_token is not None and self.unkown_token in self.mapping_dict:
                        out_str.append(self.mapping_dict[self.unkown_token])
                    else:
                        if isinstance(sample_item, str):
                            out_str.append('UNK')
                        elif isinstance(sample_item, np.ndarray):
                            out_str.append(np.random.standard_normal(sample_item.shape))
                        elif isinstance(sample_item, numbers.Integral):
                            out_str.append(0)
                        elif isinstance(sample_item, numbers.Number):
                            out_str.append(random.random())
                        else:
                            raise ValueError('{0} {1} cannot mapping properly.'.format(sample_item.__class__.__name__, sample_item))

            if is_list or self.output_type == 'list':
                return out_str
            elif self.output_type == 'array' and isinstance(out_str[0], np.ndarray):
                return np.array(out_str)
            elif self.output_type == 'string' and isinstance(out_str[0], str):
                return ''.join(out_str)
            else:
                return out_str

        #return ''.join(out_str) if not is_list else out_str

    def _apply_sequence_labels(self, labels, spec: TensorSpec):
        return labels

    def _apply_sequence_mask(self, mask, spec: TensorSpec):
        return mask

class BopomofoConvert(TextTransform):
    """
    Swap two characters in given string randomly.

    Args:
        probs (float): probability to swap characters
        name (string): transformation name.

    Examples:
        >>> cc=BopomofoConvert(convert_ratio=1)
        >>> data=list('呵呵呵，真是笑死人了，愛硬拗就會這樣下場吧。')
        >>> result=cc(data)
        >>> print(result)
        ㄏㄏㄏ，ㄓㄕ笑ㄙㄖ了，ㄞ硬ㄠ就會ㄓ樣下ㄔㄅ

    """

    def __init__(self, convert_ratio=0.4, name='bopomofo_convert', **kwargs):
        super().__init__()
        self.convert_ratio = convert_ratio
        self.text2pronounce = {}

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'text2pronounce.txt'), 'r', encoding='utf-8-sig') as f:
            for line in f.readlines():
                words = line.strip().split('\t')
                if len(words) == 3:
                    self.text2pronounce[words[0]] = words[2].split(' ')

        self.name = name

    def _apply_corpus(self, corpus, spec: TensorSpec):
        is_list = isinstance(corpus, list)
        out_str = []
        for word in corpus:
            if word in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                out_str.append(word)
            else:
                rr = random.random()
                if len(word) > 0:
                    for char in word:
                        return_char = char
                        if char in self.text2pronounce:
                            pronounce = self.text2pronounce[char]
                            if len(pronounce) > 2:
                                pronounce = pronounce[:2]
                            canconvert = all([len(p) == 2 if p[0] in ['ㄚ', 'ㄧ', 'ㄟ', 'ㄡ'] else len(p) <= 3 for p in pronounce])
                            if char in string.digits or char in string.punctuation:
                                canconvert = False
                            if any([len(p) == 2 for p in pronounce]):
                                rr = rr / 2
                            elif char in '的呵啊呃哈你他':
                                rr = rr / 2
                            elif char in '我元':
                                rr = rr * 10
                            if canconvert and rr < self.convert_ratio:
                                return_char = pronounce[0][0]
                                if char == ['呦']:
                                    return_char = 'ㄡ'
                                elif char in ['耶也']:
                                    return_char = 'ㄝ'
                        out_str.append(return_char)
        return ''.join(out_str) if not is_list else out_str

    def _apply_sequence_labels(self, labels, spec: TensorSpec):
        return labels

    def _apply_sequence_mask(self, mask, spec: TensorSpec):
        return mask


class RandomMask(TextTransform):
    """
    Masking characters or words in given string randomly.

    Args:
        probs (float): probability to swap characters
        name (string): transformation name.

    Examples:
        >>> rm1=RandomMask(convert_ratio=0.15, unit='char')
        >>> data=list('深度學習是機器學習的分支，是一種以人工神經網路為架構，對資料進行表徵學習的演算法。 ')
        >>> result=rm1(data)
        >>> print(len(result)==len(data))
        True
        >>> print(result)
        深度學習是機器學習的分支，是一種以人工神經網路為架構，對資料進行表徵學習的演算法。
        >>> rm2=RandomMask(convert_ratio=0.20, unit='word')
        >>> data=list('Deep learning is a branch of machine learning. It is an algorithm that uses artificial neural networks as an architecture to characterize and learn data.  ')
        >>> result=rm2(data)
        >>> print(len(result)==len(data))
        True
        >>> print(result)
        Deep learning is a branch of machine learning. It is an algorithm that uses artificial neural networks as an architecture to characterize and learn data.

    """

    def __init__(self, convert_ratio=0.15, unit='char', name='random_mask', **kwargs):
        super().__init__()
        self.convert_ratio = convert_ratio
        if unit not in ['word', 'char']:
            raise ValueError('Only [word,char] are valid unit option.')
        else:
            self.unit = unit
        self.name = name

    def _apply_corpus(self, corpus, spec: TensorSpec):
        is_list = isinstance(corpus, list)
        new_corpus_list = []
        if not is_list:
            corpus_list=corpus.split(' ')
            for c in corpus_list:
                new_corpus_list.append(c)
                new_corpus_list.append(' ')
            corpus=new_corpus_list[:-1]
        else:
            tmp_word=''
            for c in corpus:
                if  c in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                    if len(tmp_word)>0:
                        new_corpus_list.append(tmp_word)
                    tmp_word=''
                    new_corpus_list.append(c)
                elif c==' ' or c in string.punctuation:
                    if len(tmp_word)>0:
                        new_corpus_list.append(tmp_word)
                    tmp_word=''
                    new_corpus_list.append(c)
                else:
                    tmp_word+=c
            corpus=new_corpus_list

        out_str = []
        for word in corpus:
            if word in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                out_str.append(word)
            else:
                if self.unit == 'word':

                    if  len(word) > 1 and word != ' ' and not is_punctuation(word) and not any([s in string.digits for s in word.lower()]) and len(out_str)>2 and out_str[ -2] != '[MASK]':
                        rr=random.random()
                        if rr < self.convert_ratio:
                            for char in word:
                                out_str.append('[MASK]')
                        else:
                            out_str.extend(list(word))
                    else:
                        out_str.extend(list(word))
                else:
                    if len(word) > 0:
                        for char in word:
                            rr = random.random()
                            # if len(out_str)>2 and out_str[-2]!='[MASK]' and out_str[-1]=='[MASK]':
                            #     rr=rr*0.75
                            if rr < self.convert_ratio and char != ' ' and char not in string.punctuation and char not in string.digits:
                                # rr1 = random.random()
                                # if rr1 < 0.1:
                                #     out_str.append('[UNK]')
                                # else:
                                out_str.append('[MASK]')
                            else:
                                out_str.append(char)

        return ''.join(out_str) if not is_list else out_str

    def _apply_sequence_labels(self, labels, spec: TensorSpec):
        return labels

    def _apply_sequence_mask(self, mask, spec: TensorSpec):
        return mask


class RandomHomophonicTypo(TextTransform):
    """
    Replace by homophonic typo randomly (simulate user input using voice recognition or bopomofo/ pinyin).

    Args:
        probs (float): probability to swap characters
        name (string): transformation name.

    Examples:
        >>> ht=RandomHomophonicTypo(convert_ratio=1)
        >>> data=list('鍾先生最不齒這種沒公德心的行為，他決定挺身而出。')
        >>> result=ht(data)
        >>> print(result)
        鍾先生最不齒這種沒公德心的行為，他決定挺身而出。

    """

    def __init__(self, convert_ratio=0.5, name='random_homophonic_typo', **kwargs):
        super().__init__()
        self.convert_ratio = convert_ratio
        self.text2pronounce = OrderedDict()
        self.pronounce2text = OrderedDict()
        self.char_freq = OrderedDict()

        if not get_session().get_resources('char_freq'):
            char_freq = get_session().regist_resources('char_freq', OrderedDict())
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'char_freq.txt'), 'r', encoding='utf-8-sig') as f:
                for line in f.readlines():
                    cols = line.strip().split('\t')
                    char_freq[cols[0]] = float(cols[1])
                self.char_freq = char_freq

        else:
            self.char_freq = get_session().get_resources('char_freq')

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'text2pronounce.txt'), 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                words = line.strip().split('\t')
                if len(words) == 3:
                    prons = words[2].split(' ')
                    if 0 < len(prons) <= 2:
                        prons = prons[:1]
                    elif 5 > len(prons) > 2:
                        prons = prons[:2]
                    elif len(prons) >= 5:
                        prons = prons[:3]
                    self.text2pronounce[words[0]] = prons
                    if words[0] in self.char_freq and self.char_freq[words[0]] > -15.5:
                        for p in prons:
                            if p not in self.pronounce2text:
                                self.pronounce2text[p] = []
                            self.pronounce2text[p].append(words[0])

        self.name = name

    def _apply_corpus(self, corpus, spec: TensorSpec):
        is_list = isinstance(corpus, list)
        out_str = []
        for word in corpus:
            if word in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                out_str.append(word)
            else:
                rr = random.random()
                if len(word) > 0:
                    for char in word:
                        return_char = char
                        if char not in string.digits and char not in string.punctuation and char not in string.ascii_letters and char not in bpmf_phonetic:
                            if rr < self.convert_ratio:
                                if char in self.text2pronounce:
                                    pronounce = self.text2pronounce[char][0]
                                    candidates = self.pronounce2text[pronounce].copy() if pronounce in self.pronounce2text else []
                                    # ㄓㄔㄕㄖㄗㄘㄙ
                                    # if pronounce[0] in 'ㄓㄗ' and len(pronounce)>=3:
                                    #     for s in 'ㄓㄗ':
                                    #         if s!=pronounce[0]:
                                    #             candidates.extend(self.pronounce2text[s+pronounce[1:]].copy() if s+pronounce[1:] in self.pronounce2text else [])
                                    if pronounce[0] in 'ㄕㄘ' and len(pronounce) >= 3:
                                        for s in 'ㄕㄘ':
                                            if s != pronounce[0]:
                                                candidates.extend(self.pronounce2text[s + pronounce[1:]].copy() if s + pronounce[1:] in self.pronounce2text else [])
                                    elif pronounce[0] in 'ㄖㄙ' and len(pronounce) >= 3:
                                        for s in 'ㄖㄙ':
                                            if s != pronounce[0]:
                                                candidates.extend(self.pronounce2text[s + pronounce[1:]].copy() if s + pronounce[1:] in self.pronounce2text else [])

                                    elif pronounce[0] in 'ㄕㄖㄘㄙ':
                                        for s in 'ㄕㄖㄘㄙ':
                                            if s != pronounce[0]:
                                                candidates.extend(self.pronounce2text[s + pronounce[1:]].copy() if s + pronounce[1:] in self.pronounce2text else [])
                                    if char in candidates:
                                        candidates.remove(char)
                                    if len(candidates) == 0:
                                        pass
                                    elif len(candidates) == 1:
                                        return_char = candidates[0]
                                    elif len(candidates) > 1:
                                        max_freq = builtins.max(self.char_freq[char] - 2 if char in self.char_freq else -18, -15.5)
                                        for candidate in candidates:
                                            if candidate in self.char_freq:
                                                freq = self.char_freq[candidate]
                                                if freq > max_freq:
                                                    max_freq = freq
                                                    return_char = candidate

                            out_str.append(return_char)
                        else:
                            out_str.append(char)
        return ''.join(out_str) if not is_list else out_str

    def _apply_sequence_labels(self, labels, spec: TensorSpec):
        return labels

    def _apply_sequence_mask(self, mask, spec: TensorSpec):
        return mask


class RandomHomomorphicTypo(TextTransform):
    """
    Replace by homomorphic typo randomly (look similar, user input typo cause by eye.).

    Args:
        probs (float): probability to swap characters
        name (string): transformation name.

    Examples:
        >>> ht=RandomHomomorphicTypo(convert_ratio=1)
        >>> data=list('鍾先生最不齒這種沒公德心的行為，他決定挺身而出。')
        >>> result=ht(data)
        >>> print(result)
        鍾先生最不齒這種沒公德心的行為，他決定挺身而出。

    """

    def __init__(self, convert_ratio=0.5, name='random_homomorphic_typo', **kwargs):
        super().__init__()
        self.convert_ratio = convert_ratio
        download_file_from_google_drive('1MDk7eH7nORa16SyzNzqv7fYzBofzxGRI', dirname=os.path.join(get_trident_dir(), 'download'), filename='chardict.pkl')
        self.chardict = unpickle(os.path.join(get_trident_dir(), 'download', 'chardict.pkl'))
        self.all_embedding = to_tensor(np.stack(self.chardict.value_list, 0)).to('cpu')
        self.name = name

        if not get_session().get_resources('char_freq'):
            char_freq = get_session().regist_resources('char_freq', OrderedDict())
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'char_freq.txt'), 'r', encoding='utf-8-sig') as f:
                for line in f.readlines():
                    cols = line.strip().split('\t')
                    char_freq[cols[0]] = float(cols[1])
                self.char_freq = char_freq

        else:
            self.char_freq = get_session().get_resources('char_freq')

    def get_similar(self, char):
        if char in self.chardict and char not in string.digits and char not in string.punctuation and char not in string.ascii_letters and char not in bpmf_phonetic:
            embedding = to_tensor(expand_dims(self.chardict[char], 0)).to('cpu')
            results = element_cosine_distance(embedding, self.all_embedding, -1)[0]
            top10 = to_numpy(argsort(results, axis=0)[:5])
            results = to_numpy(results)
            similar_chars = [self.chardict.key_list[int(idx)] for idx in top10 if self.chardict.key_list[int(idx)] != char and results[int(idx)] > 0.8]
            max_freq = -15.5
            return_char = char
            for similar_char in similar_chars:
                if similar_char in self.char_freq and self.char_freq[similar_char] > max_freq:
                    max_freq = self.char_freq[similar_char]
                    return_char = similar_char
            return return_char

        else:

            return char

    def _apply_corpus(self, corpus, spec: TensorSpec):
        is_list = isinstance(corpus, list)
        out_str = []
        for word in corpus:
            if word in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                out_str.append(word)
            else:
                rr = random.random()
                if len(word) > 0:
                    for char in word:
                        return_char = char
                        if rr < self.convert_ratio:
                            out_str.append(self.get_similar(char))
                        else:
                            out_str.append(char)
        return ''.join(out_str) if not is_list else out_str

    def _apply_sequence_labels(self, labels, spec: TensorSpec):
        return labels

    def _apply_sequence_mask(self, mask, spec: TensorSpec):
        return mask