import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from math import log2
from utils.metrics import accuracy
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init, \
                                       uniform_init, xavier_init
'''
FeedForward Layer
'''
class FFN(nn.Module):
    def __init__(self, ch_in, ch_hidden, drop_rate):
        super(FFN, self).__init__()
        self.embed_dims = ch_in
        self.ffn_dims = ch_hidden
        self.fc1 = nn.Linear(self.embed_dims, self.ffn_dims)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(self.ffn_dims, self.embed_dims)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class FFN_CLS(nn.Module):
    def __init__(self, ch_in, ch_hidden, drop_rate):
        super(FFN_CLS, self).__init__()
        self.embed_dims = ch_in
        self.ffn_dims = ch_hidden
        self.fc1 = nn.Linear(self.embed_dims, self.ffn_dims)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(self.ffn_dims, self.embed_dims)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class PatchEmbed1d(nn.Module):
    def __init__(self, ch_in, embed_dim, patch_size, dropout_r):
        super(PatchEmbed1d, self).__init__()

        self.patch_embeddings = nn.Conv1d(
                in_channels=ch_in,
                out_channels=embed_dim, 
                kernel_size=patch_size,
                stride=patch_size
            )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        embeddings = self.patch_embeddings(x) # B*C,D*2 , T/2
        embeddings = embeddings.transpose(1, 2) # B*C, T/2, D*2
        embeddings = self.norm(embeddings) # only apply layer norm on the last dimesion
        embeddings = embeddings.transpose(1, 2) # B*C, D*2, T/2,
        embeddings = self.dropout(embeddings)

        return embeddings

class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_rate=0.1, drop_path_rate=0.1, is_DW_conv=True):
        super().__init__()
        if(is_DW_conv):
            group_dim = dim
        else:
            group_dim = 1
        self.pos_embed = nn.Conv1d(dim, dim, 3, padding=1, groups=group_dim)
        self.norm1 = nn.BatchNorm1d(dim)
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.attn = nn.Conv1d(dim, dim, 5, padding=2, groups=group_dim)

        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        self.norm2 = nn.BatchNorm1d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop_rate=drop_rate)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class reverse_CBlock_time(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_rate=0.1, drop_path_rate=0.1,):
        super().__init__()
        self.pos_embed = nn.Conv1d(dim, dim, 3, padding=1, groups=1)
        # self.pos_embed = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm1d(dim)
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)
        self.attn = nn.Conv1d(dim, dim, 5, padding=2, groups=1)
        # self.attn = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)

        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        self.norm2 = nn.BatchNorm1d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop_rate=drop_rate)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class reverse_CBlock_channel(nn.Module):
    def __init__(self, dim, dim_out, mlp_ratio=4., drop_rate=0.1, drop_path_rate=0.1,):
        super().__init__()
        self.pos_embed = nn.Conv1d(dim, dim_out, 3, padding=1, groups=1)
        self.norm1 = nn.BatchNorm1d(dim)
        self.conv1 = nn.Conv1d(dim, dim_out, 1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, 1)
        self.attn = nn.Conv1d(dim_out, dim_out, 5, padding=2, groups=1)

        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        self.norm2 = nn.BatchNorm1d(dim_out)
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        self.mlp = CMlp(in_features=dim_out, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop_rate=drop_rate)

    def forward(self, x):
        x = self.pos_embed(x) + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MultiheadAttention(nn.Module):
    """Multi-head Attention Module.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        attn_scale (bool): If True, use AttnScale (anti-oversmoothing).
            AttnScale decomposes a self-attention block into low-pass and
            high-pass components, then rescales and combines these two filters
            to produce an all-pass self-attention matrix.
            Defaults to False.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 ):
        super(MultiheadAttention, self).__init__()

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = DROPOUT_LAYERS.build(dropout_layer)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        # print(attn.shape)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        return x


class SABlock(BaseModule):

    def __init__(self, dim, embed_dim, ffn_ratio, num_heads, init_values, drop_rate, drop_path_rate):
        super(SABlock, self).__init__()
        
        self.embed_dims = embed_dim
        self.num_heads = num_heads
        self.init_values = init_values
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        # self.pos_embed = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.pos_embed = nn.Conv1d(dim, dim, 3, padding=1, groups=1)
        self.attn = MultiheadAttention(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            attn_drop=0.,
            proj_drop=0.,
            qkv_bias=True,)

        self.ffn = FFN(embed_dim, self.embed_dims * ffn_ratio, drop_rate)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        if self.init_values > 0:
            self.gamma_1 = nn.Parameter(
                self.init_values * torch.ones((self.embed_dims)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                self.init_values * torch.ones((self.embed_dims)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
        self.init_weights()

    def init_weights(self):
        super(SABlock, self).init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (
                nn.LayerNorm, nn.GroupNorm, nn.SyncBatchNorm)):
                constant_init(m, val=1, bias=0)
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        if self.gamma_1 is not None:
            B,C,D,T = x.shape
            x = x.view(B*C, D, T)
            x = x + self.pos_embed(x)
            x = x.view(B,C,D,T).flatten(2)
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x

class SABlock2(BaseModule):

    def __init__(self, embed_dim, num_heads):
        super(SABlock2, self).__init__()
        
        self.embed_dims = embed_dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.attn = MultiheadAttention(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            attn_drop=0.,
            proj_drop=0.,
            qkv_bias=True,)

        self.ffn = FFN(embed_dim, self.embed_dims * 2, 0.2)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=0.2))


        self.gamma_1, self.gamma_2 = None, None
    
    def init_weights(self):
        super(SABlock, self).init_weights()

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (
                nn.LayerNorm, nn.GroupNorm, nn.SyncBatchNorm)):
                constant_init(m, val=1, bias=0)
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
class PlainCNN(BaseModule):

    def __init__(self):
        super(PlainCNN, self).__init__()
                # build stem

        self.stem = nn.Conv1d(
                in_channels=84,
                out_channels=32, 
                kernel_size=15,
                stride=4,
            )
        # self.sa1 = SABlock2(8, 4)
        # self.sa2 = SABlock2(32, 8)
        # self.stem = nn.Conv1d(in_channels=16*21, out_channels=8*21, kernel_size=7, padding=3, groups=1)

        # self.layer1 = nn.Conv1d(in_channels=32*21, out_channels=32*21, kernel_size=7, padding=3, groups=1)
        # self.layer2 = nn.Conv1d(in_channels=8*21, out_channels=16*21, kernel_size=7, padding=3, groups=1)

        # self.layer3 = nn.Conv1d(in_channels=16*21, out_channels=16*21, kernel_size=5, padding=2, groups=1)
        # self.layer4 = nn.Conv1d(in_channels=32*21, out_channels=32*21, kernel_size=7, padding=3, groups=1)
        
        # self.layer1 = nn.Conv1d(in_channels=8*21, out_channels=16*21, kernel_size=7, padding=3)
        # self.layer2= nn.Conv1d(in_channels=21*16, out_channels=32*21, kernel_size=7, padding=3)
        # self.layer3= nn.Conv1d(in_channels=32*21, out_channels=32*21, kernel_size=5, padding=2)
        # self.layer4= nn.Conv1d(in_channels=32*21, out_channels=32*21, kernel_size=5, padding=2)
        self.layer1 = nn.Conv1d(
                in_channels=32,
                out_channels=64, 
                kernel_size=7,
                padding=3
            ) 
        self.BN1 = nn.BatchNorm1d(64)
        self.BN0 = nn.BatchNorm1d(32)
        
        self.layer2 = nn.Conv1d(
                in_channels=64,
                out_channels=128, 
                kernel_size=5,
                padding=2
            )
        self.BN2 = nn.BatchNorm1d(128) 

        self.layer3 = nn.Conv1d(
                in_channels=128,
                out_channels=256, 
                kernel_size=5,
                padding=2
            )
        self.BN3 = nn.BatchNorm1d(256)
        # self.layer3 = CBlock(256, mlp_ratio=2., drop_rate=0.1, drop_path_rate=0.1)

        # self.layer4 = CBlock(256, mlp_ratio=2., drop_rate=0.1, drop_path_rate=0.1)

        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pooling_stem = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):

        # print(x.shape)
        x = self.stem(x)
        x = self.BN0(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.pooling(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.pooling(x)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.pooling(x)
        x = self.BN3(x)
        x = self.relu(x)
        x = self.dropout(x)
        

        return x 

