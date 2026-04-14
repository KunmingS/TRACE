# Adapted from mmaction2/mmcv VisionTransformerAdapter — all mmcv/mmengine/mmaction deps removed.
from typing import Dict, List, Optional, Union

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Inline replacements for mmcv / mmengine utilities
# ---------------------------------------------------------------------------

def _build_norm_layer(norm_cfg: dict, num_features: int) -> nn.Module:
    """Build a norm layer from a config dict (replaces mmcv.build_norm_layer)."""
    norm_type = norm_cfg.get("type", "LN")
    eps = norm_cfg.get("eps", 1e-6)
    if norm_type == "LN":
        return nn.LayerNorm(num_features, eps=eps)
    elif norm_type == "BN":
        return nn.BatchNorm1d(num_features, eps=eps)
    elif norm_type == "BN2d":
        return nn.BatchNorm2d(num_features, eps=eps)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


def _drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic Depth per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep_prob) * mask


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


class FFN(nn.Module):
    """Feed-Forward Network (compatible with mmcv FFN state_dict naming)."""

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        act_cfg: dict = dict(type="GELU"),
        ffn_drop: float = 0.0,
        add_identity: bool = False,
    ):
        super().__init__()
        act_type = act_cfg.get("type", "GELU") if isinstance(act_cfg, dict) else "GELU"
        act = nn.GELU() if act_type == "GELU" else nn.ReLU(inplace=True)
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(embed_dims, feedforward_channels), act, nn.Dropout(ffn_drop)),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_drop),
        )
        self.add_identity = add_identity

    def forward(self, x):
        out = self.layers(x)
        if self.add_identity:
            return x + out
        return out


class PatchEmbed(nn.Module):
    """3-D Patch Embedding (replaces mmcv PatchEmbed with conv_type='Conv3d')."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 768,
        conv_type: str = "Conv3d",
        kernel_size=(2, 16, 16),
        stride=(2, 16, 16),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
    ):
        super().__init__()
        if conv_type == "Conv3d":
            self.projection = nn.Conv3d(
                in_channels, embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.projection(x)  # [B, embed_dims, T', H', W']
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, T'*H'*W', C]
        return x, (T, H, W)


def _constant_init(module: nn.Module, val: float, bias: float = 0.0):
    """Replaces mmengine constant_init."""
    nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _trunc_normal_init(module: nn.Module, std: float = 0.02, bias: float = 0.0):
    """Replaces mmengine trunc_normal_init."""
    nn.init.trunc_normal_(module.weight, std=std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def get_sinusoid_encoding(n_position: int, d_hid: int) -> Tensor:
    """Sinusoidal positional encoding (from mmaction vit_mae)."""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


# ---------------------------------------------------------------------------
# VisionTransformerAdapter classes
# ---------------------------------------------------------------------------

class Adapter(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        kernel_size: int = 3,
        dilation: int = 1,
        temporal_size: int = 384,
    ) -> None:
        super().__init__()
        hidden_dims = int(embed_dims * mlp_ratio)

        self.temporal_size = temporal_size
        self.dwconv = nn.Conv1d(
            hidden_dims, hidden_dims, kernel_size=kernel_size, stride=1,
            padding=(kernel_size // 2) * dilation, dilation=dilation, groups=hidden_dims,
        )
        self.conv = nn.Conv1d(hidden_dims, hidden_dims, 1)
        self.dwconv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / kernel_size))
        self.dwconv.bias.data.zero_()
        self.conv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / hidden_dims))
        self.conv.bias.data.zero_()

        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        _trunc_normal_init(self.down_proj, std=0.02, bias=0)
        _constant_init(self.up_proj, 0)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x
        x = self.down_proj(x)
        x = self.act(x)

        B, N, C = x.shape
        attn = x.reshape(-1, self.temporal_size, h, w, x.shape[-1])
        attn = attn.permute(0, 2, 3, 4, 1).flatten(0, 2)
        attn = self.dwconv(attn)
        attn = self.conv(attn)
        attn = attn.unflatten(0, (-1, h, w)).permute(0, 4, 1, 2, 3)
        attn = attn.reshape(B, N, C)
        x = x + attn

        x = self.up_proj(x)
        return x * self.gamma + inputs


class PlainAdapter(nn.Module):
    def __init__(self, embed_dims: int, mlp_ratio: float = 0.25, **kwargs) -> None:
        super().__init__()
        hidden_dims = int(embed_dims * mlp_ratio)
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        _trunc_normal_init(self.down_proj, std=0.02, bias=0)
        _constant_init(self.up_proj, 0)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return x * self.gamma + inputs


class Attention(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape

        if hasattr(self, "q_bias"):
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: dict = dict(type="GELU"),
        norm_cfg: dict = dict(type="LN", eps=1e-6),
        with_cp: bool = False,
        use_adapter: bool = False,
        adapter_mlp_ratio: float = 0.25,
        temporal_size: int = 384,
        **kwargs,
    ) -> None:
        super().__init__()
        self.with_cp = with_cp
        self.use_adapter = use_adapter

        self.norm1 = _build_norm_layer(norm_cfg, embed_dims)
        self.attn = Attention(
            embed_dims, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate, drop_rate=drop_rate,
        )

        self.drop_path = nn.Identity()
        if drop_path_rate > 0.0:
            self.drop_path = DropPath(drop_path_rate)
        self.norm2 = _build_norm_layer(norm_cfg, embed_dims)

        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False,
        )

        if self.use_adapter:
            self.adapter = Adapter(
                embed_dims=embed_dims,
                kernel_size=3,
                dilation=1,
                temporal_size=temporal_size,
                mlp_ratio=adapter_mlp_ratio,
            )

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        def _inner_forward(x):
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            if self.use_adapter:
                x = self.adapter(x, h, w)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VisionTransformerAdapter(nn.Module):
    """Vision Transformer with temporal adapter modules.

    Replaces mmaction VisionTransformerAdapter — no mmcv/mmengine/mmaction deps.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[int] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_cfg: dict = dict(type="LN", eps=1e-6),
        num_frames: int = 16,
        tubelet_size: int = 2,
        use_mean_pooling: bool = True,
        return_feat_map: bool = False,
        with_cp: bool = False,
        adapter_mlp_ratio: float = 0.25,
        total_frames: int = 768,
        adapter_index: list = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if adapter_index is None:
            adapter_index = [3, 5, 7, 11]

        self.with_cp = with_cp
        self.embed_dims = embed_dims
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type="Conv3d",
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
        )

        grid_size = img_size // patch_size
        num_patches = grid_size ** 2 * (num_frames // tubelet_size)
        self.grid_size = (grid_size, grid_size)

        pos_embed = get_sinusoid_encoding(num_patches, embed_dims)
        self.register_buffer("pos_embed", pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                use_adapter=i in adapter_index,
                adapter_mlp_ratio=adapter_mlp_ratio,
                temporal_size=total_frames // tubelet_size,
            )
            for i in range(depth)
        ])

        if use_mean_pooling:
            self.norm = nn.Identity()
            self.fc_norm = _build_norm_layer(norm_cfg, embed_dims)
        else:
            self.norm = _build_norm_layer(norm_cfg, embed_dims)
            self.fc_norm = None

        self.return_feat_map = return_feat_map

        num_vit_param = sum(p.numel() for name, p in self.named_parameters() if "adapter" not in name)
        num_adapter_param = sum(p.numel() for name, p in self.named_parameters() if "adapter" in name)
        ratio = num_adapter_param / num_vit_param * 100
        print("ViT params: {}, Adapter params: {}, ratio: {:2.1f}%".format(
            num_vit_param, num_adapter_param, ratio))

    def forward(self, x: Tensor) -> Tensor:
        self._freeze_layers()

        b, _, _, h, w = x.shape
        h //= self.patch_size
        w //= self.patch_size
        x = self.patch_embed(x)[0]

        if (h, w) != self.grid_size:
            pos_embed = self.pos_embed.reshape(-1, *self.grid_size, self.embed_dims)
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(h, w), mode="bicubic", align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = pos_embed.reshape(1, -1, self.embed_dims)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, h, w)

        x = self.norm(x)

        if self.return_feat_map:
            x = x.reshape(b, -1, h, w, self.embed_dims)
            x = x.permute(0, 4, 1, 2, 3)
            return x

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))

        return x[:, 0]

    def _freeze_layers(self):
        """Freeze all parameters except adapter modules."""
        self.patch_embed.eval()
        for m in self.patch_embed.modules():
            for param in m.parameters():
                param.requires_grad = False

        for block in self.blocks:
            for m, n in block.named_children():
                if "adapter" not in m and m != "drop_path":
                    n.eval()
                    for param in n.parameters():
                        param.requires_grad = False
