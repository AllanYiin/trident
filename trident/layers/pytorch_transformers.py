"""Pytorch transformer layers definition in trident"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch

from trident.backend.pytorch_backend import Layer, Sequential, get_device, ModuleList,Parameter,Tensor
from trident.backend.pytorch_ops import *
from trident.layers.pytorch_activations import Gelu
from trident.layers.pytorch_blocks import FullConnect_Block
from trident.layers.pytorch_layers import Embedding, Dropout, Dense
from trident.layers.pytorch_normalizations import LayerNorm

__all__ = ['Mlp','BERT','BERTEmbedding','PositionalEmbedding','PositionwiseFeedForward','DropPath','Attention','MultiHeadedAttention','SublayerConnection','TransformerBlock']

def Mlp(hidden_features=None, out_features=None,dropout_rate=0):
    return Sequential(
        FullConnect_Block(num_filters=hidden_features,activation=Gelu(),dropout_rate=dropout_rate,normalization=None),
        FullConnect_Block(num_filters=out_features, activation=None, dropout_rate=dropout_rate,normalization=None),
    )


class PositionalEmbedding(Layer):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float().to(get_device())
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1).to(get_device())
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp().to(get_device())

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, **kwargs):
        self.pe.to(x.device)
        return self.pe[:, :x.size(1)]

class PositionwiseFeedForward(Layer):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = Dense(num_filters= d_ff,activation=Gelu())
        self.w_2 = Dense(num_filters= d_model)
        self.dropout = Dropout(dropout_rate)


    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x)))

#
# class PositionEmbeddingSine(Layer):
#     """
#     This is a more standard version of the position embedding, very similar to the one
#     used by the Attention is all you need paper, generalized to work on images.
#     """
#     def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
#         super().__init__()
#         self.num_pos_feats = num_pos_feats
#         self.temperature = temperature
#         self.normalize = normalize
#         if scale is not None and normalize is False:
#             raise ValueError("normalize should be True if scale is passed")
#         if scale is None:
#             scale = 2 * math.pi
#         self.scale = scale
#
#     def forward(self, tensor_list):
#         x = tensor_list.tensors
#         mask = tensor_list.mask
#         assert mask is not None
#         not_mask = ~mask
#         y_embed = not_mask.cumsum(1, dtype=torch.float32)
#         x_embed = not_mask.cumsum(2, dtype=torch.float32)
#         if self.normalize:
#             eps = 1e-6
#             y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
#             x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
#
#         dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
#         dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
#
#         pos_x = x_embed[:, :, :, None] / dim_t
#         pos_y = y_embed[:, :, :, None] / dim_t
#         pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
#         return pos
#
#
# class PositionEmbeddingLearned(Layer):
#     """
#     Absolute pos embedding, learned.
#     """
#     def __init__(self, num_pos_feats=256):
#         super().__init__()
#         self.row_embed = nn.Embedding(50, num_pos_feats)
#         self.col_embed = nn.Embedding(50, num_pos_feats)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.uniform_(self.row_embed.weight)
#         nn.init.uniform_(self.col_embed.weight)
#
#     def forward(self, tensor_list):
#         x = tensor_list.tensors
#         h, w = x.shape[-2:]
#         i = torch.arange(w, device=x.device)
#         j = torch.arange(h, device=x.device)
#         x_emb = self.col_embed(i)
#         y_emb = self.row_embed(j)
#         pos = torch.cat([
#             x_emb.unsqueeze(0).repeat(h, 1, 1),
#             y_emb.unsqueeze(1).repeat(1, w, 1),
#         ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
#         return pos
#




class BERTEmbedding(Layer):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embedding_dim,cls_idx=1,sep_idx=2,unk_idx=3,pad_idx=0,mask_idx=4, dropout_rate=0.1, add_noise=False,noise_intensity=0.05):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim,padding_idx=pad_idx,add_noise=add_noise,noise_intensity=noise_intensity)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = Embedding(num_embeddings=3,embedding_dim=self.token.embedding_dim,padding_idx=0)
        self.cls_idx=cls_idx
        self.sep_idx=sep_idx
        self.pad_idx=pad_idx
        self.unk_idx=unk_idx
        self.mask_idx=mask_idx
        self.dropout_rate=dropout_rate
        self.dropout = Dropout(dropout_rate)
        self.norm=LayerNorm()
        self.embedding_dim = embedding_dim

    def forward(self, x,segments_tensor=None):
        if segments_tensor is None:
            segments_tensor = zeros_like(x).to(x.device)
            # if self.sep_idx not in x:
            #
            # else:
            #     segments_tensor_list=[]
            #     B,N=int_shape(x)
            #     sep_tuples=(x == self.sep_idx).nonzero()(as_tuple=True)
            #     for i in range(B):
            #         sep_tuple=sep_tuples[i]
            #         if len(sep_tuple)<=1:
            #             segments_tensor_list.append(zeros_like(x[i]))
            #         elif  sep_tuple==2:
            #             t=zeros_like([i]).detach()
            #             sep_tuple[:sep_tuple[0]+1]=1
            #             sep_tuple[sep_tuple[0]+1:sep_tuple[1] + 1] = 2
            #             segments_tensor_list.append(t)
            #     segments_tensor=stack(segments_tensor_list,axis=0).to(get_device())






        x = self.token(x) + self.position(x) + self.segment(segments_tensor)
        x=self.norm(x)
        if self.dropout_rate>0 and self.training:
            x=self.dropout(x)
        return x

class DropPath(Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        #return drop_path(x, self.drop_prob, self.training)

        if self.drop_prob == 0. or not self.get_root().training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class Attention(Layer):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout = Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9 if scores.dtype == torch.float32 else -1e+4)
        p_attn = softmax(scores,axis=-1)

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn



class MultiHeadedAttention(Layer):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout_rate=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = ModuleList([Dense(d_model) for _ in range(3)])
        self.output_linear = Dense(d_model)
        self.attention = Attention(dropout_rate=dropout_rate)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query=self.linear_layers[0](x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_layers[1](x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_layers[2](x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)



class SublayerConnection(Layer):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, dropout_rate=0.0):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm()
        self.dropout = DropPath(dropout_rate)


    def forward(self, x,sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class TransformerBlock(Layer):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden=None, dropout_rate=0.1):
        """
        param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        if feed_forward_hidden is None:
            feed_forward_hidden=4*hidden
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout_rate=dropout_rate)
        self.input_sublayer = SublayerConnection( dropout_rate=dropout_rate)
        self.output_sublayer = SublayerConnection(dropout_rate=dropout_rate)
        self.dropout = Dropout(dropout_rate=dropout_rate)

    def forward(self, x, mask=None):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class BERT(Layer):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout_rate=0.1,pad_idx=0):
        """
        param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.pad_idx=pad_idx
        self.dropout_rate=dropout_rate

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embedding_dim=hidden,pad_idx=self.pad_idx)
        for i in range(n_layers):
            self.add_module('transformer_block{0}'.format(i),TransformerBlock(hidden, attn_heads, hidden * 4, dropout_rate) )

    def forward(self, x,segments_tensor=None):
        if int_shape(x)[1]==2:
            x,segments_tensor=split(x,num_splits=2,axis=1)
            x=x.squeeze(1)
            segments_tensor=segments_tensor.squeeze(1)
        elif segments_tensor is None:
            segments_tensor = zeros_like(x, dtype=x.dtype).to(get_device())
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x != self.pad_idx).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segments_tensor)


        # running over multiple transformer blocks
        for name,transformer in self.named_children():
            if 'transformer_block' in name:
                x = transformer.forward(x, mask)
        return x


