from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import locale
import os
from tqdm import tqdm
from collections import *
from typing import Optional,List,Tuple
from trident.backend.common import *
from trident.backend.pytorch_ops import *
from trident.backend.pytorch_backend import to_tensor, get_device, load,fix_layer
from trident.data.utils import download_model_from_google_drive,download_file_from_google_drive
from trident.layers.pytorch_layers import *

_locale = locale.getdefaultlocale()[0].lower()

__all__ = ['Word2Vec','ChineseWord2Vec']

_trident_dir = get_trident_dir()
dirname = os.path.join(_trident_dir, 'models')
if not os.path.exists(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

download_path= os.path.join(_trident_dir, 'download','vocabs_tw.txt')
make_dir_if_need(download_path)

class Word2Vec(Embedding):
    """中文詞向量
        繼承Embedding Layer

    """

    def __init__(self, pretrained=False, locale=None, embedding_dim: Optional[int] = None, num_embeddings: Optional[int] = None, vocabs: Optional[List[str]] = None,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, filter_index=-1, keep_output: bool = False, name: Optional[str] = None) -> None:

        """
        Py Word2vec结构
        """
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse,
                         _weight=_weight, filter_index=filter_index, keep_output=keep_output, name=name)
        self.locale = _locale
        if _locale is None:
            self.locale = locale.getdefaultlocale()[0].lower()
        print('locale:', self.locale)

        self._vocabs = OrderedDict()


        download_file_from_google_drive(file_id='16yDlJJ4-O9pHF-ZbXy7XPZZk6vo3aw4e', dirname=os.path.join(_trident_dir, 'download'), filename='vocabs_tw.txt')
        with open(download_path, 'r', encoding='utf-8-sig') as f:
            vocabs_tw = f.readlines()
            vocabs_tw = [s.replace('\n', '') for s in vocabs_tw if s != '\n']
            if vocabs_tw is not None:
                for k in range(len(vocabs_tw)):
                    self._vocabs[vocabs_tw[k]] = k

            if not hasattr(self, 'tw2cn') or self.tw2cn is None:

                    self.tw2cn = OrderedDict()
                    self.cn2tw = OrderedDict()

                    for i, (w, w_cn) in tqdm(enumerate(zip(vocabs_tw, self._vocabs.keys()))):
                        if w not in self.tw2cn:
                            self.tw2cn[w] = w_cn
                        self.cn2tw[w_cn] = w

    @property
    def vocabs(self):
        # 詞彙表
        return self._vocabs

    def word2idx(self, word: str):
        # 文字轉索引(根據locale處理繁簡轉換)
        if self.locale != 'zh_cn' and word in self.tw2cn:
            word = self.tw2cn[word]
        if word in self._vocabs:
            return self._vocabs[word]
        else:
            return None

    def idx2word(self, index: int):
        # 索引轉文字(根據locale處理繁簡轉換)
        if index < len(self._vocabs):
            word = self._vocabs.key_list[index]
            if self.locale != 'zh_cn' and word in self.cn2tw:
                word = self.cn2tw[word]
            return word
        else:
            return None

    @classmethod
    def load(cls):
        # 從google drive載入模型
        st = datetime.datetime.now()
        download_model_from_google_drive('13XZPWh8QhEsC8EdIp1niLtZz0ipatSGC', dirname, 'word2vec_chinese.pth')
        recovery_model = fix_layer(load(os.path.join(dirname, 'word2vec_chinese.pth')))

        recovery_model.locale = locale.getdefaultlocale()[0].lower()
        recovery_model.to(get_device())
        download_file_from_google_drive(file_id='16yDlJJ4-O9pHF-ZbXy7XPZZk6vo3aw4e', dirname=os.path.join(_trident_dir, 'download'),filename='vocabs_tw.txt')
        if not hasattr(recovery_model, 'tw2cn') or recovery_model.tw2cn is None:
            with open(download_path, 'r', encoding='utf-8-sig') as f:
                vocabs_tw = f.readlines()
                vocabs_tw = [s.replace('\n', '') for s in vocabs_tw if s != '\n']
                recovery_model.tw2cn = OrderedDict()
                recovery_model.cn2tw = OrderedDict()

                for i, (w, w_cn) in tqdm(enumerate(zip(vocabs_tw, recovery_model._vocabs.keys()))):
                    if w not in recovery_model.tw2cn:
                        recovery_model.tw2cn[w] = w_cn
                    recovery_model.cn2tw[w_cn] = w

        et = datetime.datetime.now()
        print('total loading time:{0}'.format(et - st))
        return recovery_model

    def find_similar(self, reprt: (str, Tensor), n: int = 10, ignore_indexes=None):
        # 根據文字或是向量查詢空間中最近文字
        reprt_idx = None
        if ignore_indexes is None:
            ignore_indexes = []
        if isinstance(reprt, str):
            reprt_idx = self.word2idx(reprt)
            ignore_indexes.append(reprt_idx)
            reprt = self.weight[reprt_idx].expand_dims(0) if reprt in self._vocabs else None
        if is_tensor(reprt):
            correlate = element_cosine_distance(reprt, self.weight)[0]
            sorted_idxes = argsort(correlate, descending=True)

            sorted_idxes = sorted_idxes[:n + len(ignore_indexes)]
            sorted_idxes = to_tensor([idx for idx in sorted_idxes if idx.item() not in ignore_indexes]).long()
            probs = to_list(correlate[sorted_idxes])[:n]
            words = [self.idx2word(idx.item()) for idx in sorted_idxes][:n]
            return OrderedDict(zip(words, probs))
        else:
            raise ValueError('Valid reprt should be a word or a tensor .')

    def analogy(self, reprt1: (str, Tensor, list), reprt2: (str, Tensor, list), reprt3: (str, Tensor, list), n: int = 10):
        # 類比關係 (男人之於女人等於國王之於皇后)
        reprt1_idx = None
        reprt2_idx = None
        reprt3_idx = None
        reprt1_arr = None
        reprt2_arr = None
        reprt3_arr = None
        exclude_list = []
        if isinstance(reprt1, str):
            reprt1_idx = self.word2idx(reprt1)
            exclude_list.append(reprt1_idx)
            reprt1_arr = self.weight[reprt1_idx].expand_dims(0) if reprt1_idx is not None else None
        elif isinstance(reprt1, Tensor):
            reprt1_arr = reprt1
        elif isinstance(reprt1, list):
            if isinstance(reprt1[0], str):
                reprt1_arr = self.get_words_centroid(*reprt1)
                for item in reprt1:
                    exclude_list.append(self.word2idx(item))

        if isinstance(reprt2, str):
            reprt2_idx = self.word2idx(reprt2)
            exclude_list.append(reprt2_idx)
            reprt2_arr = self.weight[reprt2_idx].expand_dims(0) if reprt2_idx is not None else None
        elif isinstance(reprt2, Tensor):
            reprt2_arr = reprt2
        elif isinstance(reprt2, list):
            if isinstance(reprt2[0], str):
                reprt2_arr = self.get_words_centroid(*reprt2)
                for item in reprt2:
                    exclude_list.append(self.word2idx(item))

        if isinstance(reprt3, str):
            reprt3_idx = self.word2idx(reprt3)
            exclude_list.append(reprt3_idx)
            reprt3_arr = self.weight[reprt3_idx].expand_dims(0) if reprt3_idx is not None else None
        elif isinstance(reprt3, Tensor):
            reprt3_arr = reprt3
        elif isinstance(reprt3, list):
            if isinstance(reprt3[0], str):
                reprt3_arr = self.get_words_centroid(*reprt3)
                for item in reprt3:
                    exclude_list.append(self.word2idx(item))

        if reprt1_arr is not None and reprt2_arr is not None and reprt3_arr is not None:
            reprt4 = reprt2_arr - reprt1_arr + reprt3_arr
            return self.find_similar(reprt4, n=n, ignore_indexes=exclude_list)
        else:
            not_find = []
            if reprt1_arr is None:
                not_find.append(reprt1)
            if reprt2_arr is None:
                not_find.append(reprt2)
            if reprt3_arr is None:
                not_find.append(reprt3)
            raise ValueError(' ,'.join(not_find) + ' was not in vocabs.')

    def get_words_centroid(self, *args):
        # 取得數個文字的向量均值
        centroid = 0
        for arg in args:
            reprt_idx = self.word2idx(arg)
            if reprt_idx is not None:
                centroid += self.weight[reprt_idx].expand_dims(0) if reprt_idx is not None else None
        return centroid / len(args)

    def get_words_vector(self, word):
        # 取得單一文字的向量
        reprt_idx = self.word2idx(word)
        if reprt_idx is not None:
            return self.weight[reprt_idx].expand_dims(0) if reprt_idx is not None else None
        return None

    def get_enumerators(self, *args, negative_case=None, n=10, exclude_samples=True):
        # 取得整體距離輸入案例最接近，但是離負案例最遠(negative_case)的文字列表
        positive_correlate = 0
        negative_correlate = 0
        exclude_list = []
        for arg in args:
            positive_correlate += element_cosine_distance(self.get_words_vector(arg), self.weight)[0]

        correlate = positive_correlate
        if negative_case is None:
            pass
        else:
            if isinstance(negative_case, str):
                negative_case = [negative_case]
            if isinstance(negative_case, (list, tuple)):
                for arg in negative_case:
                    negative_correlate += element_cosine_distance(self.get_words_vector(arg), self.weight)[0]
                correlate = positive_correlate - negative_correlate
        sorted_idxes = argsort(correlate, descending=True)
        sorted_idxes = sorted_idxes[:n + len(exclude_list)]
        sorted_idxes = to_tensor([idx for idx in sorted_idxes if idx.item() not in exclude_list]).long()
        probs = to_list(correlate[sorted_idxes])[:n]
        words = [self.idx2word(idx.item()) for idx in sorted_idxes][:n]
        return OrderedDict(zip(words, probs))


def ChineseWord2Vec(pretrained=True, freeze_features=False, **kwargs):
    if pretrained==True:
        return Word2Vec.load()
    else:
        return Word2Vec()



