"""Pytorch transformer layers definition in trident"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from distutils.version import LooseVersion
import math
from typing import List, Tuple, Optional, Union, Callable, Any, Iterable,Iterator,Mapping, TypeVar,overload
import torch
from  torch import nn
from torch.nn import functional as f
from trident.backend.common import TensorShape
from trident.backend.pytorch_backend import Layer, Sequential, get_device, ModuleList, Parameter, Tensor,reset_name
from trident.backend.pytorch_ops import *
from trident.layers.pytorch_activations import Gelu,SquaredRelu,get_activation
from trident.layers.pytorch_blocks import FullConnect_Block
from trident.layers.pytorch_layers import Embedding, Dropout, Dense,SoftMax
from trident.layers.pytorch_normalizations import LayerNorm

__all__ = ['Mlp', 'BERT','GPT2', 'BERTEmbedding', 'PositionalEmbedding', 'PositionwiseFeedForward','ConvFeedForward', 'DropPath',
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

    def __init__(self, d_model, d_ff, activation=SquaredRelu,dropout_rate=0.1,use_bias=True):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = Dense(num_filters=d_ff, activation=get_activation(activation),use_bias=use_bias)
        self.w_2 = Dense(num_filters=d_model,use_bias=use_bias)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        x=self.dropout(self.w_2(self.w_1(x)))
        return x


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

    def __init__(self, d_model, d_ff, activation=SquaredRelu(),dropout_rate=0.1):
        super(ConvFeedForward, self).__init__()
        self.w_1 = Conv1D(num_filters=d_ff,activation=get_activation(activation))
        self.w_2 = Conv1D(num_filters=d_model)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.w_2(self.w_1(x)))


class BERTEmbedding(Layer):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embedding_dim,position_embedding_type='absolute', cls_idx=1, sep_idx=2, unk_idx=3, pad_idx=0, mask_idx=4,
                 max_seq_length=512, dropout_rate=0.1, add_noise=False, noise_intensity=0.05):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx,
                               add_noise=add_noise, noise_intensity=noise_intensity)
        self.position =Embedding(num_embeddings=max_seq_length, embedding_dim=embedding_dim, padding_idx=None)
        self.token_type = Embedding(num_embeddings=2, embedding_dim=embedding_dim, padding_idx=None)

        self.max_seq_length = max_seq_length
        self.cls_idx = cls_idx
        self.sep_idx = sep_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.mask_idx = mask_idx
        self.dropout_rate = dropout_rate
        self.dropout = Dropout(dropout_rate)
        self.norm = LayerNorm(eps=1e-12,in_sequence=True)
        self.embedding_dim = embedding_dim
        self.position_embedding_type = position_embedding_type
        self.register_buffer("position_ids", torch.arange(max_seq_length).expand((1, -1)))
        if LooseVersion(vstring=torch.__version__)>LooseVersion(vstring='1.6.0'):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(self, x, token_type_ids=None,past_key_values_length: int = 0,inputs_embeds=None):
        seq_length = x.size(1)
        position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(x.size(0), seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(int_shape(x), dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.token(x)
        token_type_embeds = self.token_type(token_type_ids)

        embeddings = inputs_embeds + token_type_embeds
        if self.position_embedding_type == "absolute":
            position_embeds  = self.position(position_ids)
            embeddings += position_embeds
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings





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



# class Attention(Layer):
#     """
#     Compute 'Scaled Dot Product Attention
#     """
#
#     def __init__(self, dropout_rate=0.1):
#         super().__init__()
#         self.dropout = Dropout(dropout_rate)
#
#
#
#     def forward(self, query, key, value, mask=None):
#         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
#
#         if mask is not None:
#             scores = scores.masked_fill(mask, -1e9 if scores.dtype == torch.float32 else -1e+4)
#         p_attn = softmax(scores, axis=-1)
#
#         if self.dropout is not None:
#             p_attn = self.dropout(p_attn)
#
#         return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(Layer):
    """
    Take in model size and number of heads.
    """

    def __init__(self, attn_heads, d_model, dropout_rate=0.1):
        super().__init__()
        assert d_model % attn_heads == 0
        self.dropout = Dropout(dropout_rate)
        # We assume d_v always equals d_k
        self.d_k = d_model // attn_heads
        self.attn_heads = attn_heads

        self.linear_layers = ModuleList([Dense(d_model) for _ in range(3)])
        self.output_linear = Dense(d_model)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.attn_heads ,self.d_k)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_mask=None,head_mask: Optional[torch.FloatTensor] = None,past_key_values: Optional[List[torch.FloatTensor]] = None,):
        batch_size = x.size(0)
        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        batch_size, seq_length, _ = x.shape



        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.transpose_for_scores(self.linear_layers[0](x))
        key = self.transpose_for_scores(self.linear_layers[1](x))
        value = self.transpose_for_scores(self.linear_layers[2](x))

        attention_scores = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(self.d_k)


        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=self.device)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None and  not (isinstance(head_mask,(tuple,list)) and all([ m is None for m in head_mask]) ):
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attn_heads * self.d_k ,)
        context_layer = context_layer.view(new_context_layer_shape)

        self_outputs = (context_layer, attention_probs)

        attention_output = self.output_linear(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class MaskedMultiHeadedAttention(Layer):
    """
    Take in model size and number of heads.
            Tensor          Type            Shape
        ===========================================================================
        q               float           (..., query_len, dims)
        k               float           (..., kv_len, dims)
        v               float           (..., kv_len, dims)
        past (*)        float           (..., past_len, dims)
        mask            bool            (..., query_len, past_len + kv_len)
        ---------------------------------------------------------------------------
        output 1        float           (..., query_len, dims)
        output 2 (*)    float           (..., past_len + kv_len, dims)
        ===========================================================================

    """

    def __init__(self, attn_heads, d_model, dropout_rate=0.1, max_seq_length=512, is_cross_attention=False,
                 layer_idx=None, use_cache=True,scale_attn_weights=True, scale_attn_by_inverse_layer_idx=False,
                 output_attentions=True):
        super().__init__()
        assert d_model % attn_heads == 0

        self.use_cache=use_cache
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

    def forward(self, x, layer_past = None,attention_mask=None, head_mask=None, encoder_hidden_states=None,encoder_attention_mask=None):
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

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if self.use_cache is True:
            present = (key, value)
        else:
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

    def __init__(self, dropout_rate=0.1,pre_norm=False):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(eps=1e-12,in_sequence=True)
        self.pre_norm=pre_norm
        self.dropout = DropPath(dropout_rate)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if self.pre_norm:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            output=sublayer(x)
            x=x + self.dropout(output if is_tensor(output) else output[0])
            return self.norm(x)


class TransformerBlock(Layer):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads,feed_forward_hidden=None, activation=Gelu(),pre_norm=False, dropout_rate=0.1):
        """
        param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        if feed_forward_hidden is None:
            feed_forward_hidden = 4 * hidden
        self.pre_norm = pre_norm
        self.attention = MultiHeadedAttention(attn_heads=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout_rate=dropout_rate,activation=activation)
        self.input_sublayer = SublayerConnection(dropout_rate=dropout_rate,pre_norm=self.pre_norm)
        self.output_sublayer = SublayerConnection(dropout_rate=dropout_rate,pre_norm=self.pre_norm)
        self.dropout = Dropout(dropout_rate=dropout_rate)

    def forward(self, x, attention_mask=None,head_mask: Optional[torch.FloatTensor] = None,past_key_values: Optional[List[torch.FloatTensor]] = None):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, attention_mask=attention_mask,head_mask=head_mask,past_key_values=past_key_values))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)



class GptTransformerBlock(Layer):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden=None,activation=Gelu(),pre_norm=False,layer_idx=None, is_cross_attention=False,dropout_rate=0.1,use_cache=True):
        """
        param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()

        if feed_forward_hidden is None:
            feed_forward_hidden = 4 * hidden
        self.use_cache=use_cache
        self.layer_idx=layer_idx
        self.pre_norm=pre_norm
        self.d_model=hidden
        self.attn_heads=attn_heads
        self.ln_1 =LayerNorm(eps=1e-5,in_sequence=True)
        self.attn = MaskedMultiHeadedAttention(attn_heads=self.attn_heads, d_model=hidden, dropout_rate=dropout_rate,layer_idx=layer_idx,use_cache=use_cache)
        self.ln_2 = LayerNorm(eps=1e-5,in_sequence=True)
        self.is_cross_attention=is_cross_attention
        if self.is_cross_attention:
            self.crossattention = MaskedMultiHeadedAttention(attn_heads=self.attn_heads, d_model=hidden, dropout_rate=dropout_rate, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn =  LayerNorm(eps=1e-5,in_sequence=True)
        else:
            self.crossattention =None
            self.ln_cross_attn =None

        self.dropout_rate=dropout_rate
        self.mlp = ConvFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout_rate=dropout_rate,activation=activation)

    def __setattr__(self, name: str, value) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Layer):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
                value.is_root = False
                for mod in value.modules():
                    if isinstance(mod, Layer) and mod.uuid != value.uuid:
                        mod.is_root = False
                reset_name(value, self._uid_prefixs)
                value.relative_name = name if not hasattr(value,
                                                          'relative_name') or value.relative_name == '' else name + '.' + value.relative_name
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch.typename(value), name))
                modules[name] = value
                value.is_root = False
                for mod in value.modules():
                    if isinstance(mod, Layer) and mod.uuid != value.uuid:
                        mod.is_root = False
                reset_name(value, self._uid_prefixs)
                value.relative_name = name if not hasattr(value,
                                                          'relative_name') or value.relative_name == '' else name + '.' + value.relative_name
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(torch.typename(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)
                    if name=='is_cross_attention' and value==True:
                        self.crossattention = MaskedMultiHeadedAttention(attn_heads=self.attn_heads, d_model=self.d_model,
                                                                         dropout_rate=self.dropout_rate,
                                                                         is_cross_attention=True, layer_idx=self.layer_idx)
                        self.ln_cross_attn = LayerNorm(eps=1e-5,in_sequence=True)



    def forward(self, x, layer_past = None, attention_mask=None, head_mask=None,encoder_hidden_states=None,encoder_attention_mask=None):
            hidden_states=x
            residual = x

            if self.pre_norm:
                hidden_states = self.ln_1(hidden_states)
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,

            )
            attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
            outputs = attn_outputs[1:]
            # residual connection
            hidden_states = attn_output + residual
            if not self.pre_norm:
                hidden_states = self.ln_1(hidden_states)

            if encoder_hidden_states is not None:
                # add one self-attention block for cross-attention
                if not hasattr(self, "crossattention"):
                    raise ValueError(
                        f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                        "cross-attention layers by setting `config.add_cross_attention=True`"
                    )
                residual = hidden_states
                if self.pre_norm:
                    hidden_states = self.ln_cross_attn(hidden_states)
                cross_attn_outputs = self.crossattention(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask

                )
                attn_output = cross_attn_outputs[0]
                # residual connection
                hidden_states = residual + attn_output
                if not self.pre_norm:
                    hidden_states = self.ln_cross_attn(hidden_states)
                outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

            residual = hidden_states
            if self.pre_norm:
                hidden_states = self.ln_2(hidden_states)

            feed_forward_hidden_states = self.mlp(hidden_states)
            # residual connection
            hidden_states = residual + feed_forward_hidden_states

            if not self.pre_norm:
                hidden_states = self.ln_2(hidden_states)

            if self.use_cache:
                outputs = (hidden_states,) + outputs
            else:
                outputs = (hidden_states,) + outputs[1:]



            return outputs  # hidden_states, present, (attentions, cross_attentions)




class BERT(Layer):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout_rate=0.1, pad_idx=0,
                 max_seq_length=512,pre_norm=False,output_mode='last_hidden_layer'):
        """
        param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        valid_output_mode=['last_hidden_layer','sum_last_4_hidden','sum_all_hidden']
        if output_mode in valid_output_mode:
            self.output_mode=output_mode
        self.max_seq_length = max_seq_length
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.pad_idx = pad_idx
        self.dropout_rate = dropout_rate
        self.num_filters = hidden
        self.pre_norm=pre_norm

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embedding_dim=hidden, pad_idx=self.pad_idx,
                                       max_seq_length=max_seq_length)
        for i in range(n_layers):
            self.add_module('transformer_block{0}'.format(i),
                            TransformerBlock(hidden, attn_heads, hidden * 4, pre_norm=pre_norm,dropout_rate=dropout_rate))
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
    def get_all_blocks(self):
        return [module for name,module in self.named_children() if 'transformer_block' in name]

    def get_head_mask(
            self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.
        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # We can specify head_mask for each layer
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask
    def forward(self, x, token_type_ids=None,attention_mask=None,head_mask: Optional[torch.FloatTensor] = None,past_key_values: Optional[List[torch.FloatTensor]] = None,):

        #batch_size, seq_length = int_shape(x)[:2]
        # if int_shape(x)[1] == 2:
        #     x, token_type_ids = split(x, num_splits=2, axis=1)
        #     x = x.squeeze(1)
        #     token_type_ids = token_type_ids.squeeze(1)

        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        head_mask = self.get_head_mask(head_mask, self.n_layers)
        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, token_type_ids=token_type_ids)

        # running over multiple transformer blocks
        outputs=[]
        for i,(name, transformer) in enumerate(self.named_children()):
            if 'transformer_block' in name:
                x = transformer.forward(x, attention_mask=attention_mask,head_mask=head_mask,past_key_values=past_key_values)
                if self.output_mode == 'sum_last_4_hidden' and i >= self.n_layers - 4:
                    outputs.append(x)
                elif self.output_mode == 'sum_all_hidden' :
                    outputs.append(x)
        if len(outputs)>0:
            x=reduce_mean(stack(outputs,axis=0),axis=0)
        return x

class GPT2(Layer):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout_rate=0.1,activation=Gelu(), pad_idx=0,
                 max_seq_length=1024,pre_norm=True,use_cache=True):
        """
        param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.max_seq_length = max_seq_length
        self.use_cache=use_cache
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.pad_idx = pad_idx
        self.dropout_rate = dropout_rate
        self.drop=Dropout(dropout_rate)
        self.num_filters = hidden
        self.norm=LayerNorm(in_sequence=True)
        self.pre_norm=pre_norm
        self.out =Dense(vocab_size, use_bias=False,activation=SoftMax(-1))

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.token = Embedding(num_embeddings=vocab_size, embedding_dim=hidden, padding_idx=pad_idx,
                               add_noise=True, noise_intensity=0.08)
        self.position =Embedding(num_embeddings=max_seq_length, embedding_dim=hidden)
        for i in range(n_layers):
            self.add_module('transformer_block{0}'.format(i),
                GptTransformerBlock(attn_heads=self.attn_heads, hidden=self.hidden,feed_forward_hidden=self.feed_forward_hidden, activation=activation,dropout_rate=dropout_rate,pre_norm=self.pre_norm,use_cache=use_cache,layer_idx=i))

    def get_all_blocks(self):
        return [module for name,module in self.named_children() if 'transformer_block' in name]

    def get_head_mask(
            self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.
        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # We can specify head_mask for each layer
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(self, x,past_key_values=None, inputs_embeds=None,position_ids=None,attention_mask=None,encoder_hidden_states=None,encoder_attention_mask=None):
        input_shape = x.size() if x is not None else inputs_embeds.size()[:-1]

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.n_layers )
        else:
            past_length = past_key_values[0][0].size(-2)



        if inputs_embeds is None:
            inputs_embeds=self.token(x)
        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1] , dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        position_embeds=self.position(position_ids)

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)



        all_hidden_states= ()
        presents = () if self.use_cache else None
        head_mask = self.get_head_mask(head_mask=None,num_hidden_layers=self.n_layers)
        for i,(block, layer_past) in enumerate(zip(self.get_all_blocks(),past_key_values)):
            # Ensure layer_past is on same device as hidden_states (might not be correct)
            if layer_past is not None:
                layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)

            outputs = block(hidden_states,layer_past=layer_past,attention_mask=attention_mask, head_mask=head_mask[i],encoder_hidden_states=encoder_hidden_states,encoder_attention_mask=encoder_attention_mask)
            hidden_states = outputs[0]
            all_hidden_states = all_hidden_states + (hidden_states,)
            if self.use_cache is True:
                presents = presents + (outputs[1],)




        hidden_states=self.norm(hidden_states)
        all_hidden_states = all_hidden_states + (hidden_states,)


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
