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
    def __init__(self, config):
        super(FFN, self).__init__()
        self.embed_dims = config.model.embed_dim
        self.ffn_dims = config.model.feedforward_channels
        self.fc1 = nn.Linear(self.embed_dims, self.ffn_dims)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.model.drop_rate)
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
    def __init__(self, config):
        super(PatchEmbed1d, self).__init__()

        self.num_patches = config.model.seq_len
        self.embed_dims = config.model.embed_dim
        self.with_cls_token = config.model.with_cls_token
        self.patch_embeddings = nn.Conv1d(
                in_channels=config.model.channels,
                out_channels=self.embed_dims, 
                kernel_size=config.model.patch_size, 
                stride=config.model.patch_size
            )
        if self.with_cls_token:
            self.num_extra_tokens = 1  # cls_token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        else:
            self.cls_token = None
            self.num_extra_tokens = 0
        
        self.pos_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + self.num_extra_tokens, self.embed_dims))
        self.dropout = nn.Dropout(config.model.drop_rate)

    def forward(self, x):
        embeddings = self.patch_embeddings(x)
        embeddings = embeddings.transpose(1, 2)
        # print("patch embedding x: ", x.shape)
        B, N, _ = x.shape
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # repeat
            x = torch.cat((cls_tokens, x), dim=1)
            embeddings = x + self.pos_embeddings[:, :(N + 1), :]
        else:
            embeddings = embeddings + self.pos_embeddings
        
        embeddings = self.dropout(embeddings)

        return embeddings

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

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        return x


class TransformerEncoderLayer(BaseModule):

    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()

        self.embed_dims = config.model.embed_dim
        self.num_heads = config.model.num_heads
        self.init_values = config.model.init_values
        self.norm1 = nn.LayerNorm(self.embed_dims, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.embed_dims, eps=1e-6)


        self.attn = MultiheadAttention(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            attn_drop=0.,
            proj_drop=0.,
            qkv_bias=True,)

        self.ffn = FFN(config)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=config.model.drop_path_rate))

        if self.init_values > 0:
            self.gamma_1 = nn.Parameter(
                self.init_values * torch.ones((self.embed_dims)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                self.init_values * torch.ones((self.embed_dims)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
    
    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (
                nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm, nn.SyncBatchNorm)):
                constant_init(m, val=1, bias=0)
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x

class MultiSubTransformer(nn.Module):
    def __init__(self, config, channels):
        super(MultiSubTransformer, self).__init__()

        self.embed_dims = config.model.embed_dim
        self.num_layers = config.model.num_layers
        self.sub_count = config.train_setting.sub_count
        self.embedding_list = ModuleList()
        # Set patch embedding with pos embedding
        for i in range(self.sub_count):
            config.model.channels = channels[i]
            self.embedding_list.append(PatchEmbed1d(config))

        # Set transformer layers
        self.layers = ModuleList()
        for i in range(self.num_layers):
            self.layers.append(TransformerEncoderLayer(config))

        # For final norm
        self.norm1 = nn.LayerNorm(self.embed_dims, eps=1e-6)
    def init_weights(self):
        super(MultiSubTransformer, self).init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (
                nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm, nn.SyncBatchNorm)):
                constant_init(m, val=1, bias=0)


    def forward(self, data):
        token_list = []
        label_list = []
        for i in range(self.sub_count):
            eeg = data[0][i].cuda()
            eeg = eeg.float()
            # print(eeg.shape)
            # The following is only for quantized data
            eeg = eeg.view(eeg.shape[0], -1, eeg.shape[3]).permute(0, 2, 1).contiguous()
            token_list.append(self.embedding_list[i](eeg)) # cls token is merged into patch embedding
            label = data[1][i].cuda() # get classification label
            label_list.append(label)

        x = torch.cat(token_list, dim=0)
        concat_labels = torch.cat(label_list, dim=0)
        for i, layer in enumerate(self.layers):
            x = layer(x)

        out = self.norm1(x)
        return out, concat_labels


class Pretrain_CODE(nn.Module):
    def __init__(self, cfg, channels=[]):
        super(Pretrain_CODE, self).__init__()

        self.poa_preditor = nn.Linear(cfg.model.embed_dim, 8)
        self.moa_preditor = nn.Linear(cfg.model.embed_dim, 6)
        self.devoice_preditor = nn.Linear(cfg.model.embed_dim, 3)
        self.asp_preditor =  nn.Linear(cfg.model.embed_dim, 3)
        self.init_preditor =  nn.Linear(cfg.model.embed_dim, 24)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.model = MultiSubTransformer(cfg, channels)
        # print(channels)


    def forward(self, data, label_eeg=None, is_train=True):
        # convert inputs from BCHW -> BHWC
        acc_list = []
        z, label = self.model(data)

        label_poa = label[:, 0] # last dim of init
        label_moa = label[:, 1]
        label_devoice = label[:, 2]
        label_asp = label[:, 3]
        label_init = label[:, 4]
        # z = z[:,1:,:] # remove cls token
        latent = F.adaptive_avg_pool1d(z.permute(0, 2, 1).contiguous(), 1).view(z.size(0), -1)

        cls_score_poa = self.poa_preditor(latent)
        cls_score_moa = self.moa_preditor(latent)
        cls_score_devoice = self.devoice_preditor(latent)
        cls_score_asp = self.asp_preditor(latent)
        cls_score_init = self.init_preditor(latent)

        acc_poa = accuracy(cls_score_poa, label_poa)
        acc_moa = accuracy(cls_score_moa, label_moa)
        acc_devoice = accuracy(cls_score_devoice, label_devoice)
        acc_asp = accuracy(cls_score_asp, label_asp)
        acc_init = accuracy(cls_score_init, label_init)
        

        total_loss = self.cls_criterion(cls_score_poa, label_poa) + self.cls_criterion(cls_score_moa, label_moa) + \
                    self.cls_criterion(cls_score_devoice, label_devoice) + self.cls_criterion(cls_score_asp, label_asp) + \
                    self.cls_criterion(cls_score_init, label_init)

        return total_loss, acc_poa, acc_moa, acc_devoice, acc_asp, acc_init, z.permute(0, 2, 1).contiguous(), label
    

class sEEG_Classifier(nn.Module):
    def __init__(self, cfg, channels=[]):
        super(sEEG_Classifier, self).__init__()

        self.init_preditor =  nn.Linear(cfg.model.embed_dim, cfg.model.num_class)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.model = MultiSubTransformer(cfg, channels)

    def forward(self, data, label_eeg=None, is_train=True):
        z, label = self.model(data)
        label_init = label[:,4]
        # z = z[:,1:,:] # remove cls token
        latent = F.adaptive_avg_pool1d(z.permute(0, 2, 1).contiguous(), 1).view(z.size(0), -1)
        cls_score = self.init_preditor(latent)
        acc = accuracy(cls_score, label_init)
        loss = self.cls_criterion(cls_score, label_init)

        return loss, acc, z.permute(0, 2, 1).contiguous(), cls_score, label