import math
import torch
import torch.nn.functional as F
from typing import Callable, List, Tuple, Union
from torch import nn
# from einops import rearrange
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_init
from fvcore.nn import FlopCountAnalysis, flop_count_table
# from timm.models.vision_transformer import Block

import numpy as np
from math import log2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import accuracy
from utils.criterion import MseIgnoreZeroLoss, SiLogLoss
from utils.train_api import sinkhorn
from models.Uniformer import *


def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        # print(f'dst shape : {dst.shape}')
        # print(f'dst_idx {dst_idx.expand(n, r, c).shape}')
        # print(f'src : {src.shape}')
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


def parse_r(
    num_layers: int, r: Union[List[int], Tuple[int, float], int], total: int = None
) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    total: The predefined total number of merged tokens.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)
    r_list = [int(min_val + step * i) for i in range(num_layers)]

    if total is not None:
        remainder = total - sum(r_list)
        if remainder != 0:
            if inflect < 0:
                r_list[0] += remainder
            else:
                r_list[-1] += remainder

    return r_list


def check_parse_r(
    num_layers: int, merge_num: int, total_num: int, r_inflect: float=0., sqrt: bool=False
):
    """
    Check the best merge ratio for the given 
    """
    gap = 1e10
    best_r = 0
    for i in range(merge_num):
        r_list = parse_r(num_layers, (i, r_inflect))
        gap_ = sum(r_list) - merge_num
        if gap > abs(gap_):
            keep_num = total_num - sum(r_list)
            if sqrt and int(keep_num ** 0.5) ** 2 != keep_num:
                continue
            best_r = i
            gap = abs(gap_)
        else:
            if gap < abs(gap_):
                break

    return int(best_r)


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

# Define a custom Conv1d for uneven groups with parallelization
class ParallelGroupedConv1d(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=5, stride=1, padding=2, preserve_dim=True):
        super(ParallelGroupedConv1d, self).__init__()
        self.groups = len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.preserve_dim = preserve_dim

        # List to store Conv1d layers for each group
        self.convs = nn.ModuleList()

        for i in range(self.groups):
            in_channels_group = self.in_channels[i]
            out_channels_group = self.out_channels[i]  # You can adjust output channels per group

            # Create Conv1d for each group with specific input/output channels
            self.convs.append(nn.Conv1d(
                in_channels=in_channels_group,
                out_channels=out_channels_group,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=1  # Each group will have its own convolution
            ))

    def forward(self, x):
        # Split input tensor into groups along the channel dimension
        split_inputs = torch.split(x, self.in_channels, dim=1)
        
        # Apply convolutions in parallel to each split
        outputs = [conv(group_input) for conv, group_input in zip(self.convs, split_inputs)]

        # Concatenate outputs along the channel dimension
        if(self.preserve_dim):
            return torch.cat(outputs, dim=1)
        else:
            return torch.stack(outputs, dim=1)

class ResBlock(nn.Module):
    def __init__(self, 
                 in_filters,
                 out_filters,
                 use_conv_shortcut = False,
                 use_agn = False,
                 ) -> None:
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut
        self.use_agn = use_agn

        if not use_agn: ## agn is GroupNorm likewise skip it if has agn before
            # self.norm1 = nn.GroupNorm(16, in_filters, eps=1e-6)
            self.norm1 = nn.BatchNorm1d(in_filters, eps=1e-5)
        # self.norm2 = nn.GroupNorm(16, out_filters, eps=1e-6)
        self.norm2 = nn.BatchNorm1d(out_filters, eps=1e-5)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv1d(in_filters, out_filters, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(out_filters, out_filters, kernel_size=3, padding=1, bias=False)

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv1d(in_filters, out_filters, kernel_size=3, padding=1, bias=False)
            else:
                self.nin_shortcut = nn.Conv1d(in_filters, out_filters, kernel_size=1, padding=0, bias=False)

    def forward(self, x, **kwargs):
        residual = x

        if not self.use_agn:
            x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual


class Attention(nn.Module):

    def __init__(
            self,
            embed_dims: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            fused_attn: bool = False,
    ) -> None:
        super().__init__()
        assert embed_dims % num_heads == 0, 'embed_dims should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = self.head_dims ** -0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dims) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dims) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(self, x: torch.Tensor, size: torch.Tensor = None):
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (qkv[0], qkv[1], qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


class SABlock(nn.Module):

    def __init__(self, embed_dim, ffn_ratio=4, num_heads=8, init_values=0., drop_rate=0., drop_path_rate=0.1):
        super(SABlock, self).__init__()
        
        self.embed_dims = embed_dim
        self.num_heads = num_heads
        self.init_values = init_values
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.attn = Attention(
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
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (
                nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, val=1.)
                nn.init.constant_(m.bias, val=0.)
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        B, C, D, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B*T, C, D)
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        x = x.reshape(B, T, C, D).permute(0, 2, 3, 1)
        return x


class ToMeSABlock(SABlock):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B*T, C, D)
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * x_attn)
        else:
            x = x + self.drop_path(x_attn)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        if self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        x = x.view(B, T, -1, D).permute(0, 2, 3, 1)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, *, embed_dim=768, num_layers=12, num_heads=12, with_cls_token=True,
                 init_values=1e-5, drop_pos_rate=0., drop_path_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.with_cls_token = with_cls_token

        if self.with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(self.embed_dim))
        else:
            self.cls_token = None
        self.pos_drop = nn.Dropout(p=drop_pos_rate)
        self.blocks = nn.Sequential(
            *[SABlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ffn_ratio=4,
                init_values=init_values,
                drop_path_rate=drop_path_rate,
            ) for _ in range(num_layers)]
        )
        # self.norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        if self.with_cls_token:
            nn.init.normal_(self.cls_token, std=1e-6)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        B, C, D, T = x.shape
        if self.with_cls_token:
            cls_tokens = self.cls_token.view(1, 1, D, 1).expand(B, 1, D, T)
            x = torch.cat((cls_tokens, x), dim=1)
            # print('add cls_token', x.shape, cls_tokens.shape)
        x = self.pos_drop(x)
        x = self.blocks(x)
        # x = self.norm(x)
        return x


def make_tome_class(transformer_class):
    class ToMeTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(
                len(self.blocks), self.r, self._tome_info["total_merge"])
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeTransformer

def apply_patch(
    model, trace_source: bool = True, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "total_merge": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, SABlock):
            module.__class__ = ToMeSABlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention


class ToMeHybridEncoder(nn.Module):
    def __init__(self, *, in_channels=1, embed_dim, ch_mult=(1, 2, 4),
                 num_res_blocks=2, num_region=32, z_channels=16,
                 num_att_blocks=2, head_dim=32, merge_ratio=None, merge_num=None,
                 with_cls_token=False, dist_head=None, num_classes=None, **ignore_kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.num_region = num_region
        self.dist_head = dist_head

        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_att_blocks = num_att_blocks
        self.with_cls_token = with_cls_token
        if isinstance(self.num_res_blocks, list):
            assert len(self.num_res_blocks) == self.num_blocks
        elif isinstance(self.num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks for i in range(self.num_blocks)]

        ## construct the model
        self.down = nn.ModuleList()
        in_ch_mult = (1,) + tuple(ch_mult)
        for i_level in range(self.num_blocks):
            down = nn.Module()
            # res blocks
            block = nn.ModuleList()
            block_dim = embed_dim * in_ch_mult[i_level]  # [1, 1, 2, 4]
            block_out = embed_dim * ch_mult[i_level]  # [1, 2, 4]
            for j in range(self.num_res_blocks[i_level]):
                block.append(ResBlock(block_out, block_out))
            down.block = block
            # downsampling
            if i_level == 0:
                down.downsample = nn.Sequential(  # T/2
                    nn.Conv1d(in_channels, block_dim, kernel_size=3, stride=2, padding=1, bias=False),
                    # nn.GroupNorm(block_dim, block_dim, eps=1e-6),
                    nn.BatchNorm1d(block_dim, eps=1e-5),
                )
            else:
                down.downsample = nn.Conv1d(block_dim, block_out, kernel_size=3, stride=2, padding=1)
            self.down.append(down)

        # middle
        self.embed_dim = block_out
        self.num_heads = int(self.embed_dim / head_dim)
        self.merge_num = merge_num
        self.attn = TransformerBlock(embed_dim=self.embed_dim, num_layers=self.num_att_blocks,
                                     num_heads=self.num_heads, with_cls_token=with_cls_token)
        self.attn = self.apply_merge(
            self.attn, merge_ratio=merge_ratio, merge_num=self.merge_num)

        # end
        self.conv_head = nn.Sequential(
            # nn.GroupNorm(16, self.embed_dim, eps=1e-6),
            nn.BatchNorm1d(self.embed_dim, eps=1e-5),
            nn.SiLU(),
            nn.Conv1d(self.embed_dim, z_channels, kernel_size=1)
        )
        if self.dist_head:  # classifer
            hidden_dim = int(4 * self.embed_dim)
            num_classes = self.embed_dim if not num_classes else num_classes
            self.dist_head = nn.Sequential(
                nn.Linear(self.embed_dim, hidden_dim, bias=True),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, num_classes, bias=True),
            )

    def forward(self, x, return_cls=False, region_indecs=None):
        ## downsampling
        B, C, D, T = x.shape
        x = x.reshape(B*C, D, T)
        for i_level in range(self.num_blocks):
            x = self.down[i_level].downsample(x)
            # print('down:', i_level, x.shape)
            for i_block in range(self.num_res_blocks[i_level]):
                x = self.down[i_level].block[i_block](x)
        x = x.reshape(B, C, self.embed_dim, -1)

        ## middle
        x = self.attn(x)
        if self.with_cls_token:
            cls_token = x[:, 0, :, :].mean(dim=-1)
            x = x[:, 1:, :, :]
        else:
            if(region_indecs):
                cls_token = x[:, region_indecs, :, :].mean(dim=[1, 3])
            else:
                cls_token = x.mean(dim=[1, 3])
        if self.dist_head is not None:
            cls_token = self.dist_head(cls_token)
        if return_cls:
            return cls_token

        ## end
        B, C, D, T = x.shape
        x = x.reshape(B*C, D, T)
        x = self.conv_head(x).reshape(B, C, -1, T)
        return (x, cls_token)

    def apply_merge(self, model, merge_ratio=None, merge_num=None):
        """ applying ToMe on the custom model """
        inflect, num_layers = -0.5, self.attn.num_layers

        # check ToMe settings
        if merge_ratio is not None:
            if isinstance(merge_ratio, tuple):
                merge_ratio, inflect = merge_ratio
            merge_num = sum(parse_r(num_layers, r=(merge_ratio, inflect)))
        elif merge_num is not None:
            merge_ratio = check_parse_r(num_layers, merge_num, self.num_region, inflect)

        # apply ToMe
        apply_patch(model)  # build ToMe class
        model.r = (merge_ratio, inflect)
        model._tome_info["total_merge"] = merge_num
        return model

    @staticmethod
    def token_unmerge(keep_tokens, source=None):
        """ recovery full tokens with the given source_matrix """
        if source is None:
            return keep_tokens
        B, C, D, T = keep_tokens.shape
        BT, C, full_C = source.shape  # [B*T, keep_C, full_C]
        keep_tokens = keep_tokens.permute(0, 3, 1, 2).reshape(BT, C, D)
        full_tokens = torch.zeros(BT, full_C, D).to(keep_tokens)
        indices = (source == 1).nonzero(as_tuple=False)

        batch_idx = indices[:, 0]
        full_tokens[batch_idx, indices[:, 2], :] = keep_tokens[batch_idx, indices[:, 1], :]
        full_tokens = full_tokens.reshape(B, T, full_C, D).permute(0, 2, 3, 1)
        return full_tokens


class ToMeHybridDecoder(nn.Module):
    def __init__(self, *, in_channels=1, embed_dim, ch_mult=(1, 2, 4),
                 num_res_blocks=2, num_region=32, z_channels=16, num_res_extra=1,
                 num_att_blocks=2, head_dim=32, num_valid_region=6, ch_per_region=31, **ignore_kwargs):
        super().__init__()

        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_att_blocks = num_att_blocks
        self.num_res_extra = num_res_extra
        self.num_region = num_region
        self.ch_per_region = ch_per_region
        self.in_channels = in_channels
        self.num_valid_region = num_valid_region
        if isinstance(num_res_blocks, list):
            assert len(self.num_res_blocks) == self.num_blocks
        elif isinstance(num_res_blocks, int):  # extra one block for Decoder
            self.num_res_blocks = [num_res_blocks + num_res_extra for i in range(self.num_blocks)]
            self.num_res_blocks[0] = self.num_res_blocks[0] - self.num_res_extra

        block_in = embed_dim * ch_mult[self.num_blocks - 1]

        self.conv_in = nn.Conv1d(z_channels, block_in, kernel_size=1, padding=0, bias=True)

        # middle
        self.embed_dim = block_in
        self.num_heads = int(self.embed_dim / head_dim)
        self.attn = TransformerBlock(embed_dim=self.embed_dim, num_layers=self.num_att_blocks,
                                     num_heads=self.num_heads, with_cls_token=False)

        # upsampling
        self.up = nn.ModuleList()
        self.adaptive = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            up = nn.Module()
            # res blocks
            block = nn.ModuleList()
            block_out = embed_dim * ch_mult[i_level]
            self.adaptive.insert(0, AdaptiveGroupNorm(z_channels, block_out))
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(ResBlock(block_out, block_out))
            up.block = block

            # downsampling
            if i_level > 0:
                block_in = embed_dim * ch_mult[i_level-1]
                up.upsample = Upsampler(block_out, block_in * 2)
            self.up.insert(0, up)

        # end
        self.conv_head = nn.Sequential(
            # nn.GroupNorm(16, block_out, eps=1e-6),
            nn.BatchNorm1d(block_out, eps=1e-5),
            nn.SiLU(),
            nn.Conv1d(block_out, in_channels * 2, kernel_size=1, bias=True)
        )
        self.conv_temporal_head = nn.Sequential(
            # nn.GroupNorm(16, block_out, eps=1e-6),
            nn.BatchNorm1d(self.num_region * self.in_channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv1d(self.num_region * in_channels, self.ch_per_region * self.num_valid_region, kernel_size=1, bias=True)
        )

    def forward(self, z):
        ## input
        B, C, D, T = z.shape
        z = z.reshape(B*C, D, T)
        style = z.clone()  # for adaptive groupnorm
        z = self.conv_in(z).reshape(B, C, -1, T)
        # print('conv in', z.shape)

        ## middle
        B, C, D, T = z.shape
        z = self.attn(z)
        z = z.reshape(B * C, D, T)

        ## upsampling
        for i_level in reversed(range(self.num_blocks)):
            ### pass in each resblock first adaGN
            z = self.adaptive[i_level](z, style)
            for i_block in range(self.num_res_blocks[i_level]):
                z = self.up[i_level].block[i_block](z)
            # print('res:', i_level, z.shape)
            if i_level > 0:
                z = self.up[i_level].upsample(z)

        ## end
        z = self.conv_head(z)
        z = depth_to_space(z, block_size=2)
        _, D, T = z.shape
        z = z.reshape(B, C, D, T)
        z = z.reshape(B, C*D, T)
        z = self.conv_temporal_head(z)
        # map back to time domain 
        return z


class Upsampler(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim * 2 if dim_out is None else dim_out
        self.conv1 = nn.Conv1d(dim, dim_out, 3, padding=1)

    def forward(self, x):
        """ input: [B C D T] """
        out = self.conv1(x)
        out = depth_to_space(out, block_size=2)
        return out


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation. """
    # check inputs
    B, D, T = x.shape
    if D % block_size != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )
    # splitting one additional dimensions from the channel dimension
    x = x.view(-1, block_size, D // block_size, T)

    # putting the new dimensions along T
    x = x.permute(0, 2, 1, 3)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(B, D // block_size, T * block_size)

    return x


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=16, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(
            num_groups=num_groups, num_channels=in_filters, eps=eps, affine=False)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps

    def forward(self, x, quantizer):
        reshape_x = False
        if x.dim() == 4:
            B, C, D, T = x.shape
            x = x.view(B*C, D, T)
            reshape_x = True
        else:
            BC, D, T = x.shape
        ### calcuate var for scale
        B, D_, T_ = quantizer.shape
        scale = quantizer.var(dim=-1) + self.eps # not unbias
        scale = scale.sqrt()
        # print('scale', scale.shape, self.gamma(scale).shape)
        scale = self.gamma(scale).view(B, D, 1)

        ### calculate mean for bias
        # bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = quantizer.mean(dim=-1)
        bias = self.beta(bias).view(B, D, 1)

        x = self.gn(x)
        x = scale * x + bias
        if reshape_x:
            return x.view(B, C, D, T)
        else:
            return x

class Hybrid_ToMe_Masked_Modeling(nn.Module):
    def __init__(self, cfg, channels=[], region_indeces=[], region_counts=[], class_weights=None, commitment_cost=0.25, decay=0.99, epsilon=1e-5, mlt_log_loss=False):
        super(Hybrid_ToMe_Masked_Modeling, self).__init__()
        self.sub_count = cfg.train_setting.sub_count
        self.use_cls = cfg.train_setting.use_cls_loss
        self.use_recon = cfg.train_setting.use_recon_loss
        self.num_cls = cfg.train_setting.cls_count
        self.use_recon = cfg.train_setting.use_recon_loss
        self.seq_len = cfg.seq_len
        self.brain_region_count = cfg.model.brain_region_count
        self.region_indeces = region_indeces
        self.region_counts = region_counts
        self.class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()
        self.padding_region_indeces = []
        self.channels = channels

        for i in range(self.sub_count):
            all_region_list = list(range(self.brain_region_count))
            self.padding_region_indeces.append([item for item in all_region_list if item not in self.region_indeces[i]])

        self.region_dim = cfg.model.region_dim
        self.channel_per_region = cfg.model.channel_per_region
        self.region_projector_list = nn.ModuleList()
        self.prediction_head_list = nn.ModuleList()
        for i in range(self.sub_count):
            projector = nn.Sequential(
                                    ParallelGroupedConv1d(self.region_counts[i], [self.region_dim // 8]*len(self.region_indeces[i]), kernel_size=7, padding=3),
                                    nn.BatchNorm1d(self.region_dim // 8 * len(self.region_indeces[i]), eps=1e-5),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.1),
                                    ParallelGroupedConv1d([self.region_dim // 8]*len(self.region_indeces[i]), [self.region_dim // 4]*len(self.region_indeces[i]), kernel_size=5, padding=2),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.BatchNorm1d(self.region_dim // 4 * len(self.region_indeces[i]), eps=1e-5),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.1),
                                    ParallelGroupedConv1d([self.region_dim // 4]*len(self.region_indeces[i]), [self.region_dim // 2]*len(self.region_indeces[i]), kernel_size=5, padding=2),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.BatchNorm1d(self.region_dim // 2 * len(self.region_indeces[i]), eps=1e-5),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.1),
                                    ParallelGroupedConv1d([self.region_dim // 2]*len(self.region_indeces[i]), [self.region_dim]*len(self.region_indeces[i]), kernel_size=5, padding=2, preserve_dim=False),
                                    )
            self.region_projector_list.append(projector)

        self.recon_head_list = nn.ModuleList()
        for i in range(self.sub_count):
            recon_projector = nn.Sequential(
                                    nn.ConvTranspose1d(len(self.region_indeces[i]) * 8, 128, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm1d(128, eps=1e-5),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.1),
                                    nn.ConvTranspose1d(128, 256, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm1d(256, eps=1e-5),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(256, self.channels[i], kernel_size=1, stride=1),
                                    )
            self.recon_head_list.append(recon_projector)
        self.drop_rate = cfg.model.drop_rate
        self.drop_path_rate = cfg.model.drop_path_rate
        self.batch_size = cfg.train_setting.opt_params.batch_size

        self.merge_index_list = []
        for i in range(self.sub_count):
            merge_index = torch.cat([torch.full((count,), i, dtype=torch.long) for i, count in enumerate(self.region_counts[i])]).reshape(self.channels[i],1)
            # Step 1: Expand B to match the shape of A
            self.merge_index_list.append(merge_index)

        # embedding layers 
        base_ch = self.region_dim
        self._embedding_dim = base_ch*8
    
        in_channels, ch, ch_mult = base_ch, base_ch*2, (1, 2, 4)
        num_att_blocks, num_res_blocks, merge_num = 4, [2, 2, 2], 0

        # build encoder
        self.encoder = ToMeHybridEncoder(in_channels=in_channels, embed_dim=ch, ch_mult=ch_mult,
                             num_res_blocks=num_res_blocks, num_region=self.brain_region_count, z_channels=16,
                             num_att_blocks=num_att_blocks, merge_num=merge_num,
                             with_cls_token=False, dist_head=None, num_classes=None)

        self.recon_criterion = nn.MSELoss()
        self.decoder_list = nn.ModuleList()

        for i in range(self.sub_count):
            decoder = ToMeHybridDecoder(in_channels=in_channels, embed_dim=ch, ch_mult=ch_mult,
                            num_res_blocks=num_res_blocks, num_region=self.brain_region_count, z_channels=16,
                            num_att_blocks=num_att_blocks, num_valid_region= len(self.region_indeces[i]), ch_per_region = 8)
            self.decoder_list.append(decoder)

        # self.regtion_mask_token = nn.Parameter(torch.zeros(1, self.region_dim, 1, 1)) 
        self.regtion_mask_token = nn.Parameter(torch.zeros(1, self.region_dim, self.seq_len // 4, self.brain_region_count))

    def forward(self, data, label_eeg=None, is_train=True):
        # convert inputs from BCHW -> BHWC   
        total_loss = 0
        total_recon_loss = 0
        total_cls_loss = 0
        input_list = []
        quantized_list = []
        public_enc_indices_list = []
        private_enc_indices_list = []
        # mask_list = []
        label_eeg_list = []
        label_list = []
        mask_list = []
        recon_list = []
        for i in range(self.sub_count):
            eeg = data[0][i].cuda()
            # mask = data[2][i].cuda()
            eeg = eeg.float() # Shape becomes [B, C, T]
            # mask = mask.float()
            label_eeg = eeg.clone().float().cuda()
            mask = data[2][i]
            label = data[1][i].cuda() # get classification label
            eeg_projected = self.region_projector_list[i](eeg)
            # These two lines are for free conv
            B,_,T = eeg_projected.shape
            eeg_pooled = eeg_projected.view(B, -1, self.region_dim, T)
            # Calculate mean of channels based on brain regions, this is to ensure learnable token captures info
            eeg_pooled = eeg_pooled.permute(0, 2, 3, 1)
            B, D, T, C = eeg_pooled.shape
            # print(eeg_projected.shape)
            # Calculate mean of channels based on brain regions, this is to ensure learnable token captures info
            # B, C, D, T = eeg_projected.shape
            B, D, T, C = eeg_pooled.shape
            print(eeg_pooled.shape)
            print(self.regtion_mask_token.shape)
            eeg_all_ch = torch.zeros(B, D, T, self.brain_region_count).cuda() # Shape [B, D, T, 21]
            region_mask_token = self.regtion_mask_token.expand(B, self.region_dim, T, self.brain_region_count) # Shape [B, D, T, 21]
            eeg_all_ch[:,:,:,self.region_indeces[i]] = eeg_pooled
            # Apply masked modeling mask with missing region mask
            mask_tensor = torch.stack(mask).cuda().permute(1, 0).to(torch.int32)
            missing_mask = torch.zeros(self.brain_region_count).to(torch.int32).cuda()
            missing_mask[self.padding_region_indeces[i]] = 1
            mask_or = mask_tensor | missing_mask
            mask_or_bool = mask_or.bool()  # Shape: [32, 21]
            # Expand the C_bool tensor to match the shape of A and B
            # The expanded shape will be [32, D, T, 21]
            mask_or_expanded = mask_or_bool.unsqueeze(1).unsqueeze(2)  # Shape: [32, 1, 1, 21]
            mask_or_expanded = mask_or_expanded.expand(B, D, T, self.brain_region_count)  # Shape: [32, D, T, 21]
            eeg_all_ch[mask_or_expanded] = region_mask_token[mask_or_expanded]
            # eeg_all_ch[:,:,:,self.padding_region_indeces[i]] = region_mask_token[:,:,:,self.padding_region_indeces[i]] # Shape [B, D, T, 21-C]
            eeg_all_ch = eeg_all_ch.permute(0, 3, 1, 2).contiguous() # Shape [B, 21, D, T]
            input_list.append(eeg_all_ch)
            mask_list.append(mask_or_bool)
            label_list.append(label)
            label_eeg_list.append(label_eeg)

        label_all = torch.cat(label_list, 0)
        x = torch.cat(input_list, 0).contiguous()
        # Encode
        B, C, D, T = x.shape
        y, cls = self.encoder(x)
        source = self.encoder.attn._tome_info['source']
        if self.encoder.with_cls_token:
            source = source[:, 1:, 1:]  # remove cls_token in source
        # encoder token unmerge
        y_full = self.encoder.token_unmerge(y, source).contiguous()
        # Apply Vq to unmerged token
        B, C_, D_, T_ = y_full.shape
        sub_tensors = torch.chunk(y_full, self.sub_count, dim=0)
        acc_list = []
        data_list = []
        for i, (sub_tensor, sub_label) in enumerate(zip(sub_tensors, label_eeg_list)):
            y_hat = self.decoder_list[i](sub_tensor)
            recon = self.recon_head_list[i](y_hat)
            recon_loss = self.recon_criterion(recon, sub_label)
            total_loss += recon_loss
            recon_list.append(recon)

        return total_loss / self.sub_count, recon_list, label_eeg_list, mask_list

class Hybrid_ToMe_ClS_convmerge(nn.Module):
    def __init__(self, cfg, channels=[], region_indeces=[], region_counts=[], class_weights=None, commitment_cost=0.25, decay=0.99, epsilon=1e-5, mlt_log_loss=False):
        super(Hybrid_ToMe_ClS_convmerge, self).__init__()
        self.sub_count = cfg.train_setting.sub_count
        self.use_cls = cfg.train_setting.use_cls_loss
        self.use_recon = cfg.train_setting.use_recon_loss
        self.num_cls = cfg.train_setting.cls_count
        self.use_recon = cfg.train_setting.use_recon_loss
        self.seq_len = cfg.seq_len
        self.brain_region_count = cfg.model.brain_region_count
        self.region_indeces = region_indeces
        self.region_counts = region_counts
        self.class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()
        self.padding_region_indeces = []
        self.channels = channels

        for i in range(self.sub_count):
            all_region_list = list(range(self.brain_region_count))
            self.padding_region_indeces.append([item for item in all_region_list if item not in self.region_indeces[i]])

        self.region_dim = cfg.model.region_dim
        self.channel_per_region = cfg.model.channel_per_region
        self.region_projector_list = nn.ModuleList()
        self.prediction_head_list = nn.ModuleList()
        for i in range(self.sub_count):
            projector = nn.Sequential(
                                    ParallelGroupedConv1d(self.region_counts[i], [self.region_dim // 8]*len(self.region_indeces[i]), kernel_size=7, padding=3),
                                    nn.BatchNorm1d(self.region_dim // 8 * len(self.region_indeces[i]), eps=1e-5),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.1),
                                    ParallelGroupedConv1d([self.region_dim // 8]*len(self.region_indeces[i]), [self.region_dim // 4]*len(self.region_indeces[i]), kernel_size=5, padding=2),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.BatchNorm1d(self.region_dim // 4 * len(self.region_indeces[i]), eps=1e-5),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.1),
                                    ParallelGroupedConv1d([self.region_dim // 4]*len(self.region_indeces[i]), [self.region_dim // 2]*len(self.region_indeces[i]), kernel_size=5, padding=2),
                                    # nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.BatchNorm1d(self.region_dim // 2 * len(self.region_indeces[i]), eps=1e-5),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.1),
                                    ParallelGroupedConv1d([self.region_dim // 2]*len(self.region_indeces[i]), [self.region_dim]*len(self.region_indeces[i]), kernel_size=5, padding=2, preserve_dim=False),
                                    )
            self.region_projector_list.append(projector)
        self.drop_rate = cfg.model.drop_rate
        self.drop_path_rate = cfg.model.drop_path_rate
        self.batch_size = cfg.train_setting.opt_params.batch_size

        self.merge_index_list = []
        for i in range(self.sub_count):
            merge_index = torch.cat([torch.full((count,), i, dtype=torch.long) for i, count in enumerate(self.region_counts[i])]).reshape(self.channels[i],1)
            # Step 1: Expand B to match the shape of A
            self.merge_index_list.append(merge_index)

        # embedding layers 
        base_ch = self.region_dim
        self._embedding_dim = base_ch*8
    
        in_channels, ch, ch_mult = base_ch, base_ch*2, (1, 2, 4)
        num_att_blocks, num_res_blocks, merge_num = 4, [2, 2, 2], cfg.model.num_merge_region

        # build encoder
        self.encoder = ToMeHybridEncoder(in_channels=in_channels, embed_dim=ch, ch_mult=ch_mult,
                             num_res_blocks=num_res_blocks, num_region=self.brain_region_count, z_channels=16,
                             num_att_blocks=num_att_blocks, merge_num=merge_num,
                             with_cls_token=False, dist_head=None, num_classes=None)
        
        # self.regtion_mask_token = nn.Parameter(torch.zeros(1, self.region_dim, 1, 1)) 
        self.regtion_mask_token = nn.Parameter(torch.zeros(1, self.region_dim, 1, self.brain_region_count))
        if(cfg.model.use_pretrain_proto):
            pre_trained_weight = torch.load(cfg.train_setting.pretrain_proto_path, map_location="cpu")
            self.regtion_mask_token.weight.data = pre_trained_weight
            self.regtion_mask_token.weight.data = self.regtion_mask_token.weight.data.float()
        if(self.use_cls):
            self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train_setting.label_smoothing, weight=self.class_weights)
            for i in range(self.sub_count):
                hidden_dim = int(4 * self._embedding_dim)
                prediction_head = nn.Sequential(
                    nn.Linear(self._embedding_dim, hidden_dim, bias=True),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, self.num_cls, bias=True),
                )
                self.prediction_head_list.append(prediction_head)

        self.mlt_log_loss = mlt_log_loss

    def forward(self, data, label_eeg=None, is_train=True):
        # convert inputs from BCHW -> BHWC   
        total_loss = 0
        total_recon_loss = 0
        total_cls_loss = 0
        input_list = []
        quantized_list = []
        public_enc_indices_list = []
        private_enc_indices_list = []
        # mask_list = []
        label_eeg_list = []
        label_list = []
        for i in range(self.sub_count):
            eeg = data[0][i].cuda()
            # mask = data[2][i].cuda()
            eeg = eeg.float() # Shape becomes [B, C, T]
            # mask = mask.float()
            label_eeg = eeg.clone().float()
            label = data[1][i].cuda() # get classification label
            eeg_projected = self.region_projector_list[i](eeg)
            # print(eeg_projected.shape)
            # Calculate mean of channels based on brain regions, this is to ensure learnable token captures info
            B, C, D, T = eeg_projected.shape
            eeg_pooled = eeg_projected
            eeg_pooled = eeg_pooled.permute(0, 2, 3, 1)
            B, D, T, C = eeg_pooled.shape
            eeg_all_ch = torch.zeros(B, D, T, self.brain_region_count).cuda() # Shape [B, D, T, 21]
            region_mask_token = self.regtion_mask_token.expand(B, self.region_dim, T, self.brain_region_count) # Shape [B, D, T, 21]
            eeg_all_ch[:,:,:,self.region_indeces[i]] = eeg_pooled
            eeg_all_ch[:,:,:,self.padding_region_indeces[i]] = region_mask_token[:,:,:,self.padding_region_indeces[i]] # Shape [B, D, T, 21-C]
            eeg_all_ch = eeg_all_ch.permute(0, 3, 1, 2).contiguous() # Shape [B, 21, D, T]
            input_list.append(eeg_all_ch)
            # mask_list.append(mask)
            label_list.append(label)
            label_eeg_list.append(label_eeg)

        label_all = torch.cat(label_list, 0)
        x = torch.cat(input_list, 0).contiguous()
        # Encode
        B, C, D, T = x.shape
        y, cls = self.encoder(x)
        source = self.encoder.attn._tome_info['source']
        if self.encoder.with_cls_token:
            source = source[:, 1:, 1:]  # remove cls_token in source
        # encoder token unmerge
        y_full = self.encoder.token_unmerge(y, source).contiguous()
        # Apply Vq to unmerged token
        B, C_, D_, T_ = y_full.shape

        sub_tensors = torch.chunk(cls, self.sub_count, dim=0)
        # If use classification
        if(self.use_cls):
            sub_labels = torch.chunk(label_all, self.sub_count, dim=0)
            acc_list = []
            data_list = []
            for i, (sub_tensor, sub_label) in enumerate(zip(sub_tensors, sub_labels)):
                # print(sub_label)
                label_init = sub_label[:, 0]
                cls_score_init = self.prediction_head_list[i](sub_tensor)

                # ACC calculation
                acc_init = accuracy(cls_score_init, label_init)
                # ACC calculation
                cls_loss = self.cls_criterion(cls_score_init, label_init)
                total_cls_loss += cls_loss
                acc_list.append(acc_init)
                data_list.append(cls_score_init)
            cls_score_all = torch.cat(data_list, 0)
        

        total_loss = total_cls_loss

        return total_loss / self.sub_count, acc_list, cls_score_all, label_all



if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    bs = 10
    ################ setup for encoder and decoder ################
    in_channels, ch, ch_mult = 2, 32, (1, 2, 4)
    num_att_blocks, num_res_blocks, merge_num = 4, [2, 2, 2], 16
    num_region, T = 32, 120  # head regions, time lengths

    # build encoder
    model = ToMeHybridEncoder(in_channels=in_channels, embed_dim=ch, ch_mult=ch_mult,
                             num_res_blocks=num_res_blocks, num_region=num_region, z_channels=16,
                             num_att_blocks=num_att_blocks, merge_num=merge_num,
                             with_cls_token=False, dist_head=None, num_classes=None)
    x = torch.randn(bs, num_region, in_channels, T)  # (B, C, D, T)
    flop = FlopCountAnalysis(model, x)
    y, cls = model(x)  # return both embedding and cls_token
    print('encoder input & output: x={}, y={}, cls={}'.format(x.shape, y.shape, cls.shape))
    if model.with_cls_token:
        source = model.attn._tome_info['source'][:, 1:, 1:]  # remove cls_token in source
    else:
        source = model.attn._tome_info['source']
    print('encoder (merge_r={}): {}, source={}'.format(model.attn.r, y.shape, source.shape))
    # print(flop_count_table(flop, max_depth=4))
    print('MACs (G) of Encoder: {:.3f}'.format(flop.total() / 1e9))

    # encoder token unmerge
    y_full = model.token_unmerge(y, source)
    print('recovery token:', y_full.shape)

    # build decoder
    model = ToMeHybridDecoder(in_channels=in_channels, embed_dim=ch, ch_mult=ch_mult,
                             num_res_blocks=num_res_blocks, num_region=num_region, z_channels=16,
                             num_att_blocks=num_att_blocks)
    flop = FlopCountAnalysis(model, y_full)
    x = model(y_full)
    # print(flop_count_table(flop, max_depth=4))
    print('decoder input & output:', y_full.shape, x.shape)
    print('MACs (G) of Decoder: {:.3f}'.format(flop.total() / 1e9))
