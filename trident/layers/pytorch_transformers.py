"""Pytorch transformer layers definition in trident"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
from  torch import nn
from torch.nn import functional as f
from trident.backend.common import TensorShape
from trident.backend.pytorch_backend import Layer, Sequential, get_device, ModuleList, Parameter, Tensor
from trident.backend.pytorch_ops import *
from trident.layers.pytorch_activations import Gelu,get_activation
from trident.layers.pytorch_blocks import FullConnect_Block
from trident.layers.pytorch_layers import Embedding, Dropout, Dense,SoftMax
from trident.layers.pytorch_normalizations import LayerNorm

__all__ = ['Mlp', 'BERT','GPT2', 'BERTEmbedding', 'PositionalEmbedding', 'PositionwiseFeedForward','ConvFeedForward', 'DropPath', 'Attention',
           'MultiHeadedAttention','MaskedMultiHeadedAttention', 'SublayerConnection', 'TransformerBlock','GptTransformerBlock']


def Mlp(hidden_features=None, out_features=None, dropout_rate=0):
    return Sequential(
        FullConnect_Block(num_filters=hidden_features, activation=Gelu(), dropout_rate=dropout_rate,
                          normalization=None),
        FullConnect_Block(num_filters=out_features, activation=None, dropout_rate=dropout_rate, normalization=None),
    )


class PositionalEmbedding(Layer):

    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        self.max_seq_length=max_seq_length
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.max_seq_length, d_model).float().to(get_device())
        pe.require_grad = False

        position = torch.arange(0, self.max_seq_length).float().unsqueeze(1).to(get_device())
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
        self.w_1 = Dense(num_filters=d_ff, activation=Gelu())
        self.w_2 = Dense(num_filters=d_model)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x)))


class Conv1D(Layer):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, num_filters, activation=None):
        super().__init__()
        self.num_filters = num_filters
        self.filter_index = -1
        self.activation = get_activation(activation)

    def build(self, input_shape: TensorShape):
        if self._built == False:
            self.input_filters = input_shape[self.filter_index]
            self.register_parameter('weight', Parameter(torch.Tensor(int(self.input_filters), self.num_filters)))
            nn.init.normal_(self.weight, std=0.02)
            self.register_parameter('bias', Parameter(torch.zeros(int(self.num_filters))))
            self._built = True

    def forward(self, x):
        size_out = x.size()[:-1] + (self.num_filters,)
        x = torch.addmm(self.bias.to(x.device), x.view(-1, x.size(-1)), self.weight.to(x.device))
        x = x.view(size_out)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvFeedForward(Layer):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(ConvFeedForward, self).__init__()
        self.w_1 = Conv1D(num_filters=d_ff, activation=Gelu())
        self.w_2 = Conv1D(num_filters=d_model)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x)))


class BERTEmbedding(Layer):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embedding_dim, cls_idx=1, sep_idx=2, unk_idx=3, pad_idx=0, mask_idx=4,
                 max_seq_length=512, dropout_rate=0.1, add_noise=False, noise_intensity=0.05):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx,
                               add_noise=add_noise, noise_intensity=noise_intensity)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_seq_length=max_seq_length)
        self.segment = Embedding(num_embeddings=3, embedding_dim=self.token.embedding_dim, padding_idx=0)
        self.max_seq_length = max_seq_length
        self.cls_idx = cls_idx
        self.sep_idx = sep_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.mask_idx = mask_idx
        self.dropout_rate = dropout_rate
        self.dropout = Dropout(dropout_rate)
        self.norm = LayerNorm(eps=1e-12)
        self.embedding_dim = embedding_dim

    def forward(self, x, segments_tensor=None):
        if segments_tensor is None:
            segments_tensor = zeros_like(x).to(x.device).detach()
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
        x = self.norm(x)
        if self.dropout_rate > 0 and self.training:
            x = self.dropout(x)
        return x


class DropPath(Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # return drop_path(x, self.drop_prob, self.training)

        if self.drop_prob == 0. or not self.get_root().training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


def prune_dense_layer(layer: Dense, index: torch.LongTensor, axis: int = 0) -> Dense:
    """
    Prune a Dense layer to keep only entries in index.
    Used to remove heads.
    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        axis (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.
    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(axis, index).clone().detach()
    b = None
    if layer.bias is not None:
        if axis == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[axis] = len(index)
    new_layer = Dense(num_filters=new_size[0], use_bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def prune_conv1d_layer(layer: Conv1D, index: torch.LongTensor, axis: int = 1) -> Conv1D:
    """
    Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
    are transposed.
    Used to remove heads.
    Args:
        layer ([`~modeling_utils.Conv1D`]): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        axis (`int`, *optional*, defaults to 1): The dimension on which to keep the indices.
    Returns:
        [`~modeling_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(axis, index).clone().detach()
    if axis == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[axis] = len(index)
    new_layer = Conv1D(num_filters=new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer



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
            scores = scores.masked_fill(mask, -1e9 if scores.dtype == torch.float32 else -1e+4)
        p_attn = softmax(scores, axis=-1)

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(Layer):
    """
    Take in model size and number of heads.
    """

    def __init__(self, attn_heads, d_model, dropout_rate=0.1):
        super().__init__()
        assert d_model % attn_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // attn_heads
        self.attn_heads = attn_heads

        self.linear_layers = ModuleList([Dense(d_model) for _ in range(3)])
        self.output_linear = Dense(d_model)
        self.attention = Attention(dropout_rate=dropout_rate)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.linear_layers[0](x).view(batch_size, -1, self.attn_heads, self.d_k).transpose(1, 2)
        key = self.linear_layers[1](x).view(batch_size, -1, self.attn_heads, self.d_k).transpose(1, 2)
        value = self.linear_layers[2](x).view(batch_size, -1, self.attn_heads, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.attn_heads * self.d_k)

        return self.output_linear(x)


class MaskedMultiHeadedAttention(Layer):
    """
    Take in model size and number of heads.
    """

    def __init__(self, attn_heads, d_model, dropout_rate=0.1, max_seq_length=512, is_cross_attention=False,
                 layer_idx=None, scale_attn_weights=True, scale_attn_by_inverse_layer_idx=False,
                 output_attentions=False):
        super().__init__()
        assert d_model % attn_heads == 0
        self.max_seq_length = max_seq_length
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_seq_length, max_seq_length), dtype=torch.uint8)).view(
                1, 1, max_seq_length, max_seq_length
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.output_attentions = output_attentions
        # We assume d_v always equals d_k
        self.d_k = d_model // attn_heads
        self.attn_heads = attn_heads
        self.d_model=d_model
        self.scale_attn_weights = scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.pruned_heads = set()
        self.is_cross_attention = is_cross_attention
        #self.split_size = (self.split_size // self.attn_heads) * (self.attn_heads - len(heads))

        if self.is_cross_attention:
            self.c_attn = Conv1D(2*self.d_model)
            self.q_attn = Conv1D(self.d_model)
        else:
            self.c_attn = Conv1D(3*self.d_model)
        self.c_proj = Conv1D(self.d_model, self.d_model)

        self.attn_dropout = Dropout(dropout_rate)
        self.resid_dropout = Dropout(dropout_rate)
        self.layer_idx = layer_idx

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self, x, attention_mask=None, head_mask=None, encoder_hidden_states=None,encoder_attention_mask=None):
        hidden_states = x
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.d_model, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.d_model, dim=2)

        query = self._split_heads(query, self.attn_heads, self.d_k)
        key = self._split_heads(key, self.attn_heads, self.d_k)
        value = self._split_heads(value, self.attn_heads, self.d_k)

        # if layer_past is not None:
        #     past_key, past_value = layer_past
        #     key = torch.cat((past_key, key), dim=-2)
        #     value = torch.cat((past_value, value), dim=-2)

        # if use_cache is True:
        #     present = (key, value)
        # else:
        present = None
        #
        # if self.reorder_and_upcast_attn:
        #     attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        # else:
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.attn_heads, self.d_k)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if self.output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class SublayerConnection(Layer):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, dropout_rate=0.0):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(eps=1e-12)
        self.dropout = DropPath(dropout_rate)

    def forward(self, x, sublayer):
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
            feed_forward_hidden = 4 * hidden
        self.attention = MultiHeadedAttention(attn_heads=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout_rate=dropout_rate)
        self.input_sublayer = SublayerConnection(dropout_rate=dropout_rate)
        self.output_sublayer = SublayerConnection(dropout_rate=dropout_rate)
        self.dropout = Dropout(dropout_rate=dropout_rate)

    def forward(self, x, mask=None):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)



class GptTransformerBlock(Layer):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden=None,layer_idx=None, is_cross_attention=False,dropout_rate=0.1):
        """
        param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()

        if feed_forward_hidden is None:
            feed_forward_hidden = 4 * hidden
        self.layer_idx=layer_idx
        self.d_model=hidden
        self.attn_heads=attn_heads
        self.ln_1 =LayerNorm(eps=1e-5)
        self.attn = MaskedMultiHeadedAttention(attn_heads=self.attn_heads, d_model=hidden, dropout_rate=dropout_rate,layer_idx=layer_idx)
        self.ln_2 = LayerNorm(eps=1e-5)
        self.is_cross_attention=is_cross_attention
        if self.is_cross_attention:
            self.crossattention = MaskedMultiHeadedAttention(attn_heads=self.attn_heads, d_model=hidden, dropout_rate=dropout_rate, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn =  LayerNorm(eps=1e-5)

        self.mlp = ConvFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout_rate=dropout_rate)


    def forward(self, x, attention_mask=None, head_mask=None,encoder_hidden_states=None,encoder_attention_mask=None):
            hidden_states=x
            residual = x
            hidden_states = self.ln_1(hidden_states)
            attn_outputs = self.attn(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                #output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
            outputs = attn_outputs[1:]
            # residual connection
            hidden_states = attn_output + residual

            if encoder_hidden_states is not None:
                # add one self-attention block for cross-attention
                if not hasattr(self, "crossattention"):
                    raise ValueError(
                        f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                        "cross-attention layers by setting `config.add_cross_attention=True`"
                    )
                residual = hidden_states
                hidden_states = self.ln_cross_attn(hidden_states)
                cross_attn_outputs = self.crossattention(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    #output_attentions=output_attentions,
                )
                attn_output = cross_attn_outputs[0]
                # residual connection
                hidden_states = residual + attn_output
                outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states)
            # residual connection
            hidden_states = residual + feed_forward_hidden_states

            # if use_cache:
            #     outputs = (hidden_states,) + outputs
            # else:
            outputs = (hidden_states,) + outputs[1:]

            return outputs  # hidden_states, present, (attentions, cross_attentions)




class BERT(Layer):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout_rate=0.1, pad_idx=0,
                 max_seq_length=512):
        """
        param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.max_seq_length = max_seq_length
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.pad_idx = pad_idx
        self.dropout_rate = dropout_rate
        self.num_filters = hidden

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embedding_dim=hidden, pad_idx=self.pad_idx,
                                       max_seq_length=max_seq_length)
        for i in range(n_layers):
            self.add_module('transformer_block{0}'.format(i),
                            TransformerBlock(hidden, attn_heads, hidden * 4, dropout_rate))
        self.decoder = Dense(num_filters=vocab_size)

    # def build(self, input_shape: TensorShape):
    #     if not self._built:
    #         #self.embedding.build(input_shape)
    #         input_shape=self.embedding.output_shape
    #         for name, transformer in self.named_children():
    #             if 'transformer_block' in name:
    #                 transformer.build(input_shape)
    #                 input_shape = transformer.output_shape
    #         self.decoder.build(input_shape)
    #         self.embedding.weight.share_memory()
    #         self.decoder.weight=self.embedding.weight
    #
    #
    #         self.to(get_device())
    #         self._built = True

    def forward(self, x, segments_tensor=None):
        if int_shape(x)[1] == 2:
            x, segments_tensor = split(x, num_splits=2, axis=1)
            x = x.squeeze(1)
            segments_tensor = segments_tensor.squeeze(1)
        elif segments_tensor is None:
            segments_tensor = zeros_like(x, dtype=x.dtype).to(get_device())
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x == self.pad_idx).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segments_tensor)

        # running over multiple transformer blocks
        for name, transformer in self.named_children():
            if 'transformer_block' in name:
                x = transformer.forward(x, mask)
        return x

class GPT2(Layer):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout_rate=0.1, pad_idx=0,
                 max_seq_length=512):
        """
        param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.max_seq_length = max_seq_length
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.pad_idx = pad_idx
        self.dropout_rate = dropout_rate
        self.drop=Dropout(dropout_rate)
        self.num_filters = hidden
        self.norm=LayerNorm()
        self.out =Dense(vocab_size, use_bias=False,activation=SoftMax())

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.token = Embedding(num_embeddings=vocab_size, embedding_dim=hidden, padding_idx=pad_idx,
                               add_noise=True, noise_intensity=0.08)
        self.position =Embedding(num_embeddings=max_seq_length, embedding_dim=hidden)
        for i in range(n_layers):
            self.add_module('transformer_block{0}'.format(i),
                GptTransformerBlock(attn_heads=self.attn_heads, hidden=self.hidden,feed_forward_hidden=self.feed_forward_hidden, dropout_rate=dropout_rate))


    def forward(self, x, inputs_embeds=None,position_ids=None):
        input_shape = x.size()
        if inputs_embeds is None:
            inputs_embeds=self.token(x)
        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1] , dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        position_embeds=self.position(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        for name, transformer in self.named_children():
            if 'transformer_block' in name:
                hidden_states = transformer(hidden_states)
        hidden_states=self.norm(hidden_states)
        logits=self.out(hidden_states)
        return logits



class Pooler(Layer):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type="cls"):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError
