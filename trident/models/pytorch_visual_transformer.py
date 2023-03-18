from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from trident import context
from trident.backend.common import TensorShape
from trident.backend.pytorch_backend import Sequential, Layer, Parameter, get_device, load, fix_layer
from trident.backend.pytorch_ops import int_shape, meshgrid, expand_dims, softmax, sqrt, reshape
from trident.data.utils import download_model_from_google_drive
from trident.data.vision_transforms import Resize, Normalize
from trident.layers.pytorch_activations import Gelu, Identity
from trident.layers.pytorch_blocks import ShortCut, For, FullConnect_Block
from trident.layers.pytorch_initializers import trunc_normal
from trident.layers.pytorch_layers import Dense, Dropout, Conv2d, SoftMax
from trident.layers.pytorch_normalizations import LayerNorm
from trident.models.pretrained_utils import _make_recovery_model_include_top
from trident.optims.pytorch_trainer import ImageClassificationModel

__all__ = ['VisionTransformer_small', 'VisionTransformer', 'PatchToImage', 'ImageToPatch', 'Block', 'PatchEmbed',
           'PositionEmbed', 'Attention', 'TransformerUpsampling']
ctx = context._context()
_device = get_device()
_epsilon = ctx.epsilon
_trident_dir = ctx.trident_dir

dirname = os.path.join(_trident_dir, 'models')
if not os.path.exists(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass


def Mlp(hidden_features=None, out_features=None, drop=0):
    return Sequential(
        FullConnect_Block(num_filters=hidden_features, activation=Gelu(), dropout_rate=drop, normalization=None,
                          use_bias=True),
        FullConnect_Block(num_filters=out_features, activation=None, dropout_rate=drop, normalization=None,
                          use_bias=True),
    )


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


class Attention(Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Dense(num_filters=dim * 3, use_bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(num_filters=dim, use_bias=qkv_bias)
        self.proj_drop = Dropout(proj_drop)

    def forward(self, x):
        B, N, C = int_shape(x)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# class Block(Layer):
#
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0):
#         super().__init__()
#
#         self.norm1 = LayerNorm()
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
#         self.norm2 = LayerNorm()
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp( hidden_features=mlp_hidden_dim, out_features=dim, drop=drop)
#
#     def forward(self, x):
#
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

def Block(dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0):
    mlp_hidden_dim = int(dim * mlp_ratio)
    return Sequential(
        ShortCut(
            Identity(),
            Sequential(
                LayerNorm(in_sequence=True),
                Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                          proj_drop=drop),
                DropPath(drop_prob=drop_path)
            ), mode='add'
        ),
        ShortCut(
            Identity(),
            Sequential(
                LayerNorm(in_sequence=True),
                Mlp(hidden_features=mlp_hidden_dim, out_features=dim, drop=drop),
                DropPath(drop_prob=drop_path)
            ), mode='add'
        )
    )


class PatchEmbed(Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=16, embed_dim=None, name='patch_embed'):
        super().__init__(name=name)
        self.embed_dim = 3 * patch_size * patch_size if embed_dim is None else embed_dim
        self.patch_size = patch_size
        self.proj = Conv2d((patch_size, patch_size), num_filters=self.embed_dim, strides=patch_size, use_bias=True)

    def build(self, input_shape: TensorShape):
        if not self._built:
            b, c, height, width = input_shape.dims
            self.input_filters = int(input_shape[self.filter_index])
            self.num_patches = (width // self.patch_size) * (height // self.patch_size)
            self._built = True

    def forward(self, x):
        # transpose=>B N C
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(Layer):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, patch_size=16, embed_dim=768, feature_size=None, name='patch_embed'):
        super().__init__(name=name)
        self.patch_size = patch_size
        self.backbone = backbone
        self.proj = Conv2d((patch_size, patch_size), num_filters=embed_dim, strides=patch_size)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PositionEmbed(Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, mode='trainable', drop_rate=0, use_cls_token=True, name='pos_embed'):
        super().__init__(name=name)
        self.mode = mode
        self.use_cls_token = use_cls_token
        self.pos_drop = Dropout(dropout_rate=drop_rate)

    def build(self, input_shape: TensorShape):
        # B, 196,768
        if not self._built:
            B, N, C = input_shape.dims
            if self.use_cls_token:
                self.cls_token = Parameter(torch.zeros(1, 1, C)).to(get_device())
                trunc_normal(self.cls_token, std=.02)
                self.pos_embed = Parameter(torch.zeros(1, N + 1, C)).to(get_device())
            else:
                self.pos_embed = Parameter(torch.zeros(1, N, C)).to(get_device())

            if self.mode == 'trainable':
                trunc_normal(self.pos_embed, std=.02)
            elif self.mode == 'meshgrid':
                self.pos_embed = Parameter(
                    expand_dims(meshgrid(N + 1, C, normalized_coordinates=True, requires_grad=True).mean(-1), 0).to(
                        get_device()))

            self._built = True

    def forward(self, x):
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x


class TransformerUpsampling(Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, scale_factor=2, mode='pixel_shuffle', name=None, keep_output=False, **kwargs):
        super(TransformerUpsampling, self).__init__(keep_output=keep_output, name=name)
        self.scale_factor = scale_factor
        self.mode = mode
        self.fc = None

    def build(self, input_shape: TensorShape):
        if not self._built:
            B, N, C = input_shape.dims
            self.fc = Dense(C * self.scale_factor * self.scale_factor)

    def forward(self, x):
        B, N, C = x.size()
        x = self.fc(x)
        # assert N == H * W
        H = W = int(sqrt(N))
        x = x.permute(0, 2, 1)
        x = x.view(-1, C * self.scale_factor * self.scale_factor, H, W)
        x = nn.PixelShuffle(self.scale_factor)(x)
        B, C, H, W = x.size()
        x = x.view(-1, C, H * W)
        x = x.permute(0, 2, 1)
        return x


#
# class VisionTransformer(Layer):
#     """ Vision Transformer
#     A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
#         https://arxiv.org/abs/2010.11929
#     """
#     def __init__(self, patch_size=16, num_classes=1000, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None):
#         """
#         Args:
#             img_size (int, tuple): input image size
#             patch_size (int, tuple): patch size
#             in_chans (int): number of input channels
#             num_classes (int): number of classes for classification head
#             embed_dim (int): embedding dimension
#             depth (int): depth of transformer
#             num_heads (int): number of attention heads
#             mlp_ratio (int): ratio of mlp hidden dim to embedding dim
#             qkv_bias (bool): enable bias for qkv if True
#             qk_scale (float): override default qk scale of head_dim ** -0.5 if set
#             representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
#             drop_rate (float): dropout rate
#             attn_drop_rate (float): attention dropout rate
#             drop_path_rate (float): stochastic depth rate
#             hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
#             norm_layer: (nn.Module): normalization layer
#         """
#         super().__init__()
#
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = patch_size*patch_size*3  # num_features for consistency with other models
#         norm_layer = LayerNorm(eps=1e-6)
#
#         if hybrid_backbone is not None:
#             self.patch_embed = HybridEmbed(hybrid_backbone)
#         else:
#             self.patch_embed = PatchEmbed(patch_size=patch_size)
#
#
#         self.pos_drop = Dropout(dropout_rate=drop_rate)
#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.blocks =ModuleList([
#             Block(
#                 dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
#             for i in range(depth)])
#         self.norm = LayerNorm(eps=1e-6)
#
#         # Representation layer
#         if representation_size:
#             self.num_features = representation_size
#             self.pre_logits = Sequential(OrderedDict([
#                 ('fc', Dense(num_filters=representation_size)),
#                 ('act', Tanh())
#             ]))
#         else:
#             self.pre_logits = Identity()
#
#         # Classifier head
#         self.head = Dense( num_classes) if num_classes > 0 else Identity()
#         self.softmax=SoftMax()
#
#
#
#     def _init_weights(self, m):
#         if isinstance(m, Dense):
#             trunc_normal(m.weight, std=.02)
#             if isinstance(m, Dense) and m.bias is not None:
#                 fill_zeros(m.bias)
#         # elif isinstance(m, LayerNorm):
#         #     fill_zeros(m.bias)
#         #     fill_ones(m.weight)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}
#
#     def get_classifier(self):
#         return self.head
#
#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = Dense(num_classes) if num_classes > 0 else Identity()
#
#     def build(self, input_shape: TensorShape):
#         if not self._built:
#             for modual in self.modules():
#                 if 'Conv' not in  modual.__class__.__name__:
#                     modual.filter_index=-1
#                     modual.in_sequence=True
#             self.patch_embed.build(input_shape)
#             #num_patches = self.patch_embed.num_patches
#             self.cls_token = Parameter(torch.zeros(1, 1, self.embed_dim)).to(get_device())
#             self.pos_embed = Parameter(torch.zeros(1,  self.patch_embed.num_patches + 1, self.embed_dim)).to(get_device())
#             trunc_normal(self.pos_embed, std=.02)
#             trunc_normal(self.cls_token, std=.02)
#
#             # self.apply(self._init_weights)
#
#     def forward_features(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)  #(None,196,768)
#
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x) # #(None,197,768)
#         for blk in self.blocks:
#             x = blk(x)
#         # (None,197,768)
#         x = self.norm(x)[:, 0]  #(None,768)
#         x = self.pre_logits(x)
#         return x
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         x=self.softmax(x)
#         return x

class PatchToImage(Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=16, name=None, keep_output=False, **kwargs):
        super(PatchToImage, self).__init__(keep_output=keep_output, name=name)
        self.patch_size = patch_size

    def forward(self, x):
        # x = depth_to_space(x,block_size= self.patch_size)
        B, embed_dim, H, W = x.size()
        if embed_dim == self.patch_size * self.patch_size * 3:
            x = x.view(B, self.patch_size, self.patch_size, 3, H, W)
            x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
            x = reshape(x, (B, 3, self.patch_size * H, self.patch_size * W))
        return x


class ImageToPatch(Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=16, name=None, keep_output=False, **kwargs):
        super(ImageToPatch, self).__init__(keep_output=keep_output, name=name)
        self.patch_size = patch_size

    def forward(self, x):
        kernel = self.patch_size
        stride = self.patch_size
        dilation = 1
        b, c, h, w = x.shape
        h2 = math.ceil(h / stride)
        w2 = math.ceil(w / stride)
        pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
        pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
        x = F.pad(x, (pad_row // 2, pad_row - pad_row // 2, pad_col // 2, pad_col - pad_col // 2))

        # Extract patches
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()

        patches = patches.view(b, -1, patches.shape[-2], patches.shape[-1])
        return patches.flatten(2).transpose(1, 2)


class VisionTransformer(Layer):
    def __init__(self, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., include_top=True,
                 num_classes=1000, global_pool='token',
                 qkv_bias=True, qk_scale=768 ** -0.5, class_token=True, fc_norm=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.,
                 name='vit'):
        super().__init__(name=name)
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        self.include_top = include_top
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 if class_token else 0

        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim, name='patch_embed')
        # self.patch_embed.input_shape = TensorShape([None] + [3, 224, 224])
        self.pos_embed = PositionEmbed(drop_rate=0, mode='trainable')
        # self.pos_embed.input_shape = TensorShape( [None] + [self.patch_embed.num_patches, self.embed_dim])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = Sequential(
            For(range(depth), lambda i:
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
                )
        )
        self.norm = LayerNorm(in_sequence=True)
        self.fc_norm = LayerNorm(in_sequence=True) if use_fc_norm else Identity()
        if self.include_top:
            self.head = Dense(num_filters=self.num_classes, activation=SoftMax(), use_bias=True)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        if self.global_pool:
            x = x[:, self.num_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        if self.include_top:
            return self.head(x)
        else:
            return x


# class DistilledVisionTransformer(VisionTransformer):
#     """ Vision Transformer with distillation token.
#     Paper: `Training data-efficient image transformers & distillation through attention` -
#         https://arxiv.org/abs/2012.12877
#     This impl of distilled ViT is taken from https://github.com/facebookresearch/deit
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dist_token =Parameter(torch.zeros(1, 1, self.embed_dim))
#         num_patches = self.patch_embed.num_patches
#         self.pos_embed = Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
#         self.head_dist = Dense(self.num_classes ) if self.num_classes > 0 else Identity()
#
#         trunc_normal(self.dist_token, std=.02)
#         trunc_normal(self.pos_embed, std=.02)
#         self.head_dist.apply(self._init_weights)
#
#     def forward_features(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)
#
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         dist_token = self.dist_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, dist_token, x), dim=1)
#
#         x = x + self.pos_embed
#         x = self.pos_drop(x)
#
#         for blk in self.blocks:
#             x = blk(x)
#
#         x = self.norm(x)
#         return x[:, 0], x[:, 1]
#
#     def forward(self, x):
#         x, x_dist = self.forward_features(x)
#         x = self.head(x)
#         x_dist = self.head_dist(x_dist)
#         if self.training:
#             return x, x_dist
#         else:
#             # during inference, return the average of both classifier predictions
#             return (x + x_dist) / 2
#


def VisionTransformer_small(pretrained=False, input_shape=(3, 224, 224), patch_size=16, embed_dim=384,
                            freeze_features=True, include_top=True, num_classes=1000, depth=12, num_heads=6,
                            drop_rate=0.2, **kwargs):
    """ My custom 'small' ViT model. Depth=8, heads=8= mlp_ratio=4."""

    vit = VisionTransformer(patch_size=patch_size, embed_dim=embed_dim, include_top=include_top,
                            num_classes=num_classes, depth=depth,
                            num_heads=num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                            drop_rate=drop_rate, attn_drop_rate=drop_rate, drop_path_rate=drop_rate)
    vitmodel = ImageClassificationModel(input_shape=input_shape, output=vit)

    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        model_filename = None
        if embed_dim == 384 and input_shape == (3, 224, 224) and patch_size == 16:
            model_filename = 'vit_small_patch16_224.pth'
            download_model_from_google_drive('1FVMp83WyCNpdD_9zo98CyvqSdZlVJE-Z', dirname, 'vit_small_patch16_224.pth')
        recovery_model = load(os.path.join(dirname, model_filename))
        recovery_model = fix_layer(recovery_model)
        recovery_model = _make_recovery_model_include_top(recovery_model, input_shape=input_shape,
                                                          include_top=include_top, classes=num_classes,
                                                          freeze_features=freeze_features)
        vitmodel.model = recovery_model

        if num_classes != 1000:
            vitmodel.model.head = Dense(num_classes, activation=SoftMax(-1))
        else:
            if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_labels1.txt')):
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_labels1.txt'), 'r',
                          encoding='utf-8-sig') as f:
                    labels = [l.rstrip() for l in f]
                    vitmodel.class_names = labels
    vitmodel.preprocess_flow = [Resize((input_shape[1], input_shape[2]), keep_aspect=True), Normalize(0, 255),
                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return vitmodel