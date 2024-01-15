import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import logging
from functools import partial
from collections import OrderedDict
from transformers import ViTImageProcessor, ViTForImageClassification



import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_
batch_size = 16




class LocalModel(nn.Module):
    def __init__(self, base, predictor):
        super(LocalModel, self).__init__()

        self.base = base
        self.predictor = predictor
        
    def forward(self, x):
        out = self.base(x)
        out = self.predictor(out)

        return out


class LocalModel_pt(nn.Module):
    def __init__(self, generator, base):
        super(LocalModel_pt, self).__init__()
        self.generator = generator
        self.base = base


    def forward(self, x):
        out = self.generator(x)
        out = self.base(out)

        return out



class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

# ====================================================================================================================

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class PadPrompter(nn.Module):
    def __init__(self,inchanel = 1,pad_size=1,image_size=28,args =None):
        super(PadPrompter, self).__init__()
        pad_size = pad_size
        image_size = image_size
        self.args = args
        self.inchanel =inchanel
        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, self.inchanel, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, self.inchanel, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, self.inchanel, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1,self.inchanel, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, self.inchanel, self.base_size, self.base_size).to(self.args.device)


        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt



# ====================================================================================================================
class PatchEmbed(nn.Module):  # PatchEmbed from timm
    """
    Image to Patch Embedding
    CNN_proj + Rearrange: [B, C, W, H] -> [B, D, Pn_W, Pn_H] -> [B, D, Pn] -> [B, Pn, D]
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # CNN_proj + Rearrange: [B, C, W, H] -> [B, D, Pn_W, Pn_H] -> [B, D, Pn] -> [B, Pn, D]
        x = self.proj(x).flatten(2).transpose(1, 2)

        # x: (B, 14*14, 768)
        return x


class FFN(nn.Module):  # Mlp from timm
    """
    FFN (from timm)
    :param in_features:
    :param hidden_features:
    :param out_features:
    :param act_layer:
    :param drop:
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x


# MHSA
class Attention(nn.Module):  # qkv Transform + MSA(MHSA) (Attention from timm)
    """
    qkv Transform + MSA(MHSA) (from timm)
    # input  x.shape = batch, patch_number, patch_dim
    # output  x.shape = batch, patch_number, patch_dim
    :param dim: dim=CNN feature dim, because the patch size is 1x1
    :param num_heads:
    :param qkv_bias:
    :param qk_scale: by default head_dim ** -0.5  (squre root)
    :param attn_drop: dropout rate after MHSA
    :param proj_drop:
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # input x.shape = batch, patch_number, patch_dim
        batch, patch_number, patch_dim = x.shape

        # mlp transform + head split [B, Pn, D] -> [B, Pn, 3D] -> [B, Pn, 3, H, D/H] -> [3, B, H, Pn, D/H]
        qkv = self.qkv(x).reshape(batch, patch_number, 3, self.num_heads, patch_dim //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # 3 [B, H, Pn, D/H]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # [B, H, Pn, D/H] -> [B, H, Pn, D/H]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)  # Dropout

        # head fusion: [B, H, Pn, D/H] -> [B, Pn, H, D/H] -> [B, Pn, D]
        x = (attn @ v).transpose(1, 2).reshape(batch, patch_number, patch_dim)

        x = self.proj(x)
        x = self.proj_drop(x)  # mlp

        # output x.shape = batch, patch_number, patch_dim
        return x


# Encoder_Block
class Block(nn.Module):  # teansformer Block from timm

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        # input x.shape = batch, patch_number, patch_dim
        # output x.shape = batch, patch_number, patch_dim
        :param dim: dim
        :param num_heads:
        :param mlp_ratio: FFN
        :param qkv_bias:
        :param qk_scale: by default head_dim ** -0.5  (squre root)
        :param drop:
        :param attn_drop: dropout rate after Attention
        :param drop_path: dropout rate after sd
        :param act_layer: FFN act
        :param norm_layer: Pre Norm
        """
        super().__init__()
        # Pre Norm
        self.norm1 = norm_layer(dim)  # Transformer used the nn.LayerNorm
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE from timm: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # stochastic depth

        # Add & Norm
        self.norm2 = norm_layer(dim)

        # FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # MHSA + Res_connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # FFN + Res_connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class VisionTransformer(nn.Module):  # From timm
    """
    Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=128, depth=8,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Embedding tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic Depth Decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Encoders
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        # last norm
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.fc = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.head_dist = None

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pre_logits(x[:, 0])  # use cls token for cls head [B,1,768]
        x = self.fc(x)
        return x
