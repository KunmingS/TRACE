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
    """Bottleneck adapter with a depthwise mixing conv.

    ``mode="temporal"`` (default, original behaviour): depthwise Conv1d over the
    temporal token axis (per spatial location) — captures motion.
    ``mode="spatial"``: depthwise Conv2d over the H×W grid (per timestep) —
    captures appearance/layout. Used by the AST-proxy adaptive layout where
    shallow layers get spatial adapters and deep layers get larger-kernel
    temporal adapters. Default args reproduce the original temporal adapter
    exactly, so existing configs are unaffected.
    """

    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        kernel_size: int = 3,
        dilation: int = 1,
        temporal_size: int = 384,
        mode: str = "temporal",
    ) -> None:
        super().__init__()
        hidden_dims = int(embed_dims * mlp_ratio)

        self.temporal_size = temporal_size
        self.mode = mode
        if mode == "temporal":
            self.dwconv = nn.Conv1d(
                hidden_dims, hidden_dims, kernel_size=kernel_size, stride=1,
                padding=(kernel_size // 2) * dilation, dilation=dilation, groups=hidden_dims,
            )
            self.conv = nn.Conv1d(hidden_dims, hidden_dims, 1)
            fan = kernel_size
        elif mode == "spatial":
            self.dwconv = nn.Conv2d(
                hidden_dims, hidden_dims, kernel_size=kernel_size, stride=1,
                padding=(kernel_size // 2) * dilation, dilation=dilation, groups=hidden_dims,
            )
            self.conv = nn.Conv2d(hidden_dims, hidden_dims, 1)
            fan = kernel_size * kernel_size
        else:
            raise ValueError(f"Adapter mode must be 'temporal' or 'spatial', got {mode!r}")
        self.dwconv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan))
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
        if self.mode == "temporal":
            attn = x.reshape(-1, self.temporal_size, h, w, C)
            attn = attn.permute(0, 2, 3, 4, 1).flatten(0, 2)   # [B*h*w, C, t]
            attn = self.dwconv(attn)
            attn = self.conv(attn)
            attn = attn.unflatten(0, (-1, h, w)).permute(0, 4, 1, 2, 3)
            attn = attn.reshape(B, N, C)
        else:  # spatial: depthwise conv over H×W per timestep
            attn = x.reshape(-1, self.temporal_size, h, w, C)
            attn = attn.permute(0, 1, 4, 2, 3).flatten(0, 1)   # [B*t, C, h, w]
            attn = self.dwconv(attn)
            attn = self.conv(attn)
            attn = attn.unflatten(0, (-1, self.temporal_size)).permute(0, 1, 3, 4, 2)
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
        adapter_mode: str = "temporal",
        adapter_kernel: int = 3,
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
                kernel_size=adapter_kernel,
                dilation=1,
                temporal_size=temporal_size,
                mlp_ratio=adapter_mlp_ratio,
                mode=adapter_mode,
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
        adapter_layout: str = None,
        ast_split: int = None,
        ast_temporal_kernel: int = 7,
        unfreeze_last_n: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        if adapter_index is None:
            adapter_index = [3, 5, 7, 11]
        # Per-layer adapter (mode, kernel) assignment.
        # Default: homogeneous temporal k3 (original behaviour).
        # adapter_layout="ast_proxy": shallow layers (< ast_split, default depth//2)
        #   use spatial k3 adapters (appearance), deep layers use temporal large-kernel
        #   (ast_temporal_kernel) adapters (motion) — a cheap AST-Adapter proxy.
        split = ast_split if ast_split is not None else depth // 2
        def _layer_adapter(i):
            if adapter_layout == "ast_proxy":
                return ("spatial", 3) if i < split else ("temporal", ast_temporal_kernel)
            return ("temporal", 3)

        self.with_cp = with_cp
        self.unfreeze_last_n = unfreeze_last_n
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
                adapter_mode=_layer_adapter(i)[0],
                adapter_kernel=_layer_adapter(i)[1],
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
        """Freeze patch_embed + the ViT blocks, EXCEPT adapter modules and the
        last ``unfreeze_last_n`` transformer blocks.

        ``unfreeze_last_n=0`` (default) reproduces the original frozen-body +
        trainable-adapter (PEFT) behaviour exactly. ``unfreeze_last_n=N`` keeps
        the final N blocks trainable for end-to-end fine-tuning (=depth ⇒ all
        blocks train). patch_embed (conv stem) stays frozen; the final norm /
        fc_norm are never touched here so they stay trainable. Adapters always
        stay trainable regardless of N.
        """
        self.patch_embed.eval()
        for m in self.patch_embed.modules():
            for param in m.parameters():
                param.requires_grad = False

        train_from = len(self.blocks) - max(0, int(self.unfreeze_last_n))
        for i, block in enumerate(self.blocks):
            block_trainable = i >= train_from
            for m, n in block.named_children():
                if "adapter" in m or m == "drop_path":
                    continue  # adapters always trainable; drop_path has no params
                if block_trainable:
                    continue  # leave this block's submodules trainable
                n.eval()
                for param in n.parameters():
                    param.requires_grad = False


# ── merged from backbone_wrapper.py ──
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm


BACKBONE_MAP = {
    "VisionTransformerAdapter": VisionTransformerAdapter,
}


def _simple_compose(transforms_cfg, pipelines_registry):
    """Build a pipeline from a list of transform configs.

    Returns both the callable runner and the instantiated transforms. Most
    existing transforms are stateless callables, but spatial head experiments
    can install ``nn.Module`` transforms in the post-processing pipeline. Those
    modules must be registered on ``BackboneWrapper`` so their parameters are
    visible to the optimizer, EMA, and state_dict.
    """
    if not transforms_cfg:
        return None, None
    transforms = [pipelines_registry.build(t) for t in transforms_cfg]

    def _run(results):
        for t in transforms:
            results = t(results)
        return results

    return _run, transforms


class BackboneWrapper(nn.Module):
    """Wraps a video backbone (e.g. VisionTransformerAdapter).

    Replaces the mmaction.Recognizer3D wrapper. Handles:
    - Direct instantiation of the backbone (no mmengine registry)
    - Loading pretrained weights via torch.load
    - Pixel-level normalization (formerly ActionDataPreprocessor)
    - Pre/post-processing pipelines
    - Temporal checkpointing for memory efficiency
    """

    def __init__(self, cfg):
        super().__init__()

        # Support both dict and ConfigDict
        cfg = dict(cfg)
        custom_cfg = cfg.pop("custom")
        if hasattr(custom_cfg, "__getitem__"):
            custom_cfg = dict(custom_cfg)
        else:
            custom_cfg = custom_cfg

        # ----------------------------------------------------------------
        # Build the backbone (wrapped in a holder to match TRACEv2 naming)
        # In TRACEv2, self.model = Recognizer3D which has self.backbone = ViT
        # So param names are model.backbone.blocks.X.* — this makes the
        # optimizer exclude=["backbone"] filter work correctly.
        # ----------------------------------------------------------------
        backbone_type = cfg.pop("type")
        if backbone_type not in BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone type: '{backbone_type}'. "
                f"Available: {list(BACKBONE_MAP.keys())}"
            )
        self.model = nn.Module()
        self.model.backbone = BACKBONE_MAP[backbone_type](**cfg)

        # ----------------------------------------------------------------
        # Normalization (was: ActionDataPreprocessor)
        # Default: ImageNet mean/std used by VideoMAE
        # ----------------------------------------------------------------
        mean = custom_cfg.get("mean", [123.675, 116.28, 103.53])
        std = custom_cfg.get("std", [58.395, 57.12, 57.375])
        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).reshape(1, 1, 3, 1, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).reshape(1, 1, 3, 1, 1, 1),
        )

        # ----------------------------------------------------------------
        # Load pretrained weights
        # ----------------------------------------------------------------
        pretrain = custom_cfg.get("pretrain", None)
        if pretrain is not None:
            self._load_pretrained(pretrain)
        else:
            print(
                "Warning: no pretrain path provided — backbone will be randomly initialised "
                "unless weights are loaded elsewhere."
            )

        # ----------------------------------------------------------------
        # Pre/post processing pipelines
        # ----------------------------------------------------------------
        pre_pipeline_cfg = custom_cfg.get("pre_processing_pipeline", None)
        post_pipeline_cfg = custom_cfg.get("post_processing_pipeline", None)

        if pre_pipeline_cfg or post_pipeline_cfg:
            # Lazy import to avoid circular imports
            from vtrace.datasets.builder import PIPELINES
            self.pre_processing_pipeline, pre_transforms = _simple_compose(pre_pipeline_cfg, PIPELINES)
            self.post_processing_pipeline, post_transforms = _simple_compose(post_pipeline_cfg, PIPELINES)
            self._pipeline_modules = nn.ModuleList(
                [t for t in (pre_transforms or []) if isinstance(t, nn.Module)]
                + [t for t in (post_transforms or []) if isinstance(t, nn.Module)]
            )
        else:
            self.pre_processing_pipeline = None
            self.post_processing_pipeline = None
            self._pipeline_modules = nn.ModuleList()

        # ----------------------------------------------------------------
        # Misc settings
        # ----------------------------------------------------------------
        self.norm_eval = custom_cfg.get("norm_eval", True)
        self.freeze_backbone = custom_cfg.get("freeze_backbone", False)
        trainable_patterns = custom_cfg.get("trainable_patterns", None)
        if trainable_patterns is not None:
            trainable_patterns = tuple(trainable_patterns)
            trainable_params = 0
            frozen_params = 0
            for name, param in self.named_parameters():
                if not name.startswith("model.backbone."):
                    continue
                if any(pattern in name for pattern in trainable_patterns):
                    param.requires_grad = True
                    trainable_params += param.numel()
                else:
                    param.requires_grad = False
                    frozen_params += param.numel()
            print(
                "backbone trainable_patterns="
                f"{list(trainable_patterns)}, trainable_params={trainable_params}, frozen_params={frozen_params}"
            )
        print(f"freeze_backbone: {self.freeze_backbone}, norm_eval: {self.norm_eval}")

        self.use_temporal_checkpointing = custom_cfg.get("temporal_checkpointing", False)
        if self.use_temporal_checkpointing:
            self.temporal_checkpointing_chunk_num = custom_cfg[
                "temporal_checkpointing_chunk_num"
            ]
            self.temporal_checkpointing_chunk_dim = custom_cfg[
                "temporal_checkpointing_chunk_dim"
            ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, frames, masks=None, return_feat_map=False):
        """Forward pass.

        Args:
            frames: Tensor of shape [B, num_segs, C, T, H, W]
            masks:  Tensor of shape [B, T] (bool) or None
            return_feat_map: if True, return the ViT spatial feature map
                ``[B, C, T_tok, h, w]`` (chunk-rearranged to the full temporal
                token axis, num_segs mean-reduced) INSTEAD of the pooled
                ``[B, C, T]``. Used by detector-side spatial fusion so the H/W grid
                survives to be fused with the (D) detail / (F) frame-context
                branches before pooling. Skips the configured
                ``post_processing_pipeline`` (ConvSpatialPool / Reduce + temporal
                interpolation) — the caller owns pooling + interpolation.
        """
        self.set_norm_layer()

        # original (pre-chunk) batch size, needed to undo the chunk fold below
        orig_batch = frames.shape[0]

        # Normalise: (pixel - mean) / std
        frames = (frames.float() - self.mean) / self.std

        # Pre-processing pipeline
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        # Flatten batch × num_segs
        batches, num_segs = frames.shape[0:2]
        frames = frames.flatten(0, 1).contiguous()

        # Go through backbone
        if self.freeze_backbone:
            with torch.no_grad():
                if self.use_temporal_checkpointing:
                    features = self._temporal_checkpointing(
                        frames,
                        self.temporal_checkpointing_chunk_num,
                        self.temporal_checkpointing_chunk_dim,
                    )
                else:
                    features = self.model.backbone(frames)
        else:
            if self.use_temporal_checkpointing:
                features = self._temporal_checkpointing(
                    frames,
                    self.temporal_checkpointing_chunk_num,
                    self.temporal_checkpointing_chunk_dim,
                )
            else:
                features = self.model.backbone(frames)

        if return_feat_map:
            if isinstance(features, (tuple, list)):
                raise NotImplementedError("return_feat_map expects a single feature tensor")
            # ViT feat map: [batches*num_segs, C, T_tok, h, w] where batches =
            # orig_batch * chunk_num (the pre-processing folded T into the batch).
            feats = features.unflatten(dim=0, sizes=(batches, num_segs))  # [b*t1, n, C, t, h, w]
            feats = feats.mean(dim=1)  # mean over num_segs -> [b*t1, C, t, h, w]
            t1 = batches // orig_batch  # chunk_num
            feats = feats.unflatten(dim=0, sizes=(orig_batch, t1))  # [b, t1, C, t, h, w]
            b, _, c, t, h, w = feats.shape
            feats = feats.permute(0, 2, 1, 3, 4, 5).reshape(b, c, t1 * t, h, w)  # [b, C, t1*t, h, w]
            return feats.to(torch.float32)

        # Unflatten and pool
        if isinstance(features, (tuple, list)):
            features = torch.cat(
                [self._unflatten_and_pool(f, batches, num_segs) for f in features],
                dim=1,
            )
        else:
            features = self._unflatten_and_pool(features, batches, num_segs)

        # Apply mask
        if masks is not None and features.dim() == 3:
            features = features * masks.unsqueeze(1).detach().float()

        return features.to(torch.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unflatten_and_pool(self, features, batches, num_segs):
        features = features.unflatten(dim=0, sizes=(batches, num_segs))
        if self.post_processing_pipeline is not None:
            features = self.post_processing_pipeline(dict(feats=features))["feats"]
        return features

    def set_norm_layer(self):
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm, _BatchNorm)):
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def _load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights with flexible key matching."""
        from vtrace.weights import resolve as _resolve_weights
        checkpoint_path = _resolve_weights(checkpoint_path)
        print(f"Loading pretrained backbone from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Extract state dict from various checkpoint formats
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Strip common prefixes, repeatedly. A full-model checkpoint nests the ViT
        # two levels down (`backbone.model.backbone.blocks...`); peeling once leaves
        # `model.backbone.*`, which matches nothing and loads zero keys.
        while True:
            for prefix in ("backbone.", "model.backbone.", "module.backbone.", "module."):
                stripped = {
                    k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
                }
                if len(stripped) > 0 and len(stripped) >= len(state_dict) // 2:
                    state_dict = stripped
                    break
            else:
                break

        # Remap old-style pretrained keys to match model naming
        remapped = {}
        for k, v in state_dict.items():
            k = k.replace("patch_embed.proj.", "patch_embed.projection.")
            k = k.replace(".mlp.fc1.", ".mlp.layers.0.0.")
            k = k.replace(".mlp.fc2.", ".mlp.layers.1.")
            remapped[k] = v
        state_dict = remapped

        # Tubelet inflation/deflation: the model's patch_embed Conv3d may use a
        # different temporal kernel than the pretrained one (e.g. tubelet 2 -> 1 to
        # get one feature per frame). Adapt the temporal axis (dim=2) by mean-pooling
        # (deflate) or temporally interpolating (inflate) so the spatial filter is
        # preserved. strict=False would otherwise raise on the size mismatch.
        pe_key = "patch_embed.projection.weight"
        own = self.model.backbone.state_dict()
        if pe_key in state_dict and pe_key in own:
            w_pre, w_own = state_dict[pe_key], own[pe_key]
            if w_pre.shape != w_own.shape and w_pre.shape[:2] == w_own.shape[:2] and w_pre.shape[3:] == w_own.shape[3:]:
                t_pre, t_own = w_pre.shape[2], w_own.shape[2]
                if t_own == 1:                      # deflate: average the temporal taps
                    state_dict[pe_key] = w_pre.mean(dim=2, keepdim=True)
                else:                               # inflate/resample temporal dim
                    state_dict[pe_key] = F.interpolate(
                        w_pre.permute(0, 1, 3, 4, 2).reshape(-1, w_pre.shape[3] * w_pre.shape[4], t_pre),
                        size=t_own, mode="linear", align_corners=False,
                    ).reshape(w_pre.shape[0], w_pre.shape[1], w_pre.shape[3], w_pre.shape[4], t_own).permute(0, 1, 4, 2, 3)
                print(f"  patch_embed temporal kernel adapted {t_pre} -> {t_own} (tubelet change)")

        missing, unexpected = self.model.backbone.load_state_dict(state_dict, strict=False)

        # These keys are expected to be missing from pretrained weights:
        # - pos_embed: computed via sinusoidal encoding at init
        # - adapter.*: TRACE-specific modules, randomly initialized
        # - fc_norm.*: initialized at model construction
        expected_missing = {"pos_embed", "fc_norm.weight", "fc_norm.bias"}
        real_missing = [k for k in missing
                        if k not in expected_missing and "adapter" not in k]

        loaded = len(state_dict) - len(unexpected)
        print(f"  Loaded {loaded}/{len(state_dict)} pretrained keys.")
        if real_missing:
            print(f"  WARNING — unexpected missing keys ({len(real_missing)}): {real_missing[:5]}{'...' if len(real_missing)>5 else ''}")
        if unexpected:
            print(f"  WARNING — unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    def _temporal_checkpointing(self, frames, chunk_num, chunk_dim):
        """Memory-efficient temporal checkpointing."""
        def _inner(f):
            return self.model.backbone(f)

        chunks = torch.chunk(frames, chunk_num, dim=chunk_dim)
        video_feat = [cp.checkpoint(_inner, chunk, use_reentrant=False) for chunk in chunks]

        if isinstance(video_feat[0], (tuple, list)):
            return [
                torch.cat([f[idx] for f in video_feat], dim=chunk_dim)
                for idx in range(len(video_feat[0]))
            ]
        return torch.cat(video_feat, dim=chunk_dim)


# ── merged from vjepa2_backbone.py ──
"""Trainable V-JEPA 2 ViT-L backbone for the TRACE DenseLocalizer (end-to-end fine-tuning).

Reuses the V-JEPA 2 loader and preprocessing of the frozen-feature path, but makes
the whole encoder DIFFERENTIABLE so gradients flow into V-JEPA 2 during fine-tuning.

Model: transformers.VJEPA2Model.from_pretrained("facebook/vjepa2-vitl-fpc32-256-diving48",
local_files_only=True). Only the 24-layer ViT-L *encoder* is used; the diving48
pooler/classifier and the JEPA `predictor` are dropped (not constructed / not in the
optimizer). 256px crop, ImageNet normalization (handled upstream by BackboneWrapper-style
mean/std baked to the 0-255 scale, see config), tubelet 2 -> 16 token-frames per 32-frame
window, patch 16 -> 256 spatial tokens, hidden 1024.

Temporal alignment (identical to the frozen run): a 768-frame clip is tiled into
contiguous 32-frame windows (768/32 = 24 windows). Each window -> [16, 1024] after
spatial mean-pool. The 24 windows concatenate to 384 token-frames, then a differentiable
linear interpolation upsamples to T=768. Output: [B, 1024, 768] (channel-first), exactly
what SGPPyramidProj (in_channels=1024) expects.

Memory: end-to-end ViT-L over 24 windows/clip with backprop is heavy. We run all windows
of a clip as one batch through the encoder and rely on HF gradient checkpointing
(`gradient_checkpointing_enable`) on the 24 transformer blocks, plus bf16/AMP + bs1 +
grad-accum from the solver. Optional `freeze_first_n_blocks` zeroes-out (freezes) early
blocks to cut activation memory; layer-decay LR in the optimizer further tapers low layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class SpatialAttentionPool(nn.Module):
    """FERAL-style learnable-query attention pooling over the S spatial tokens of
    each token-frame (replaces spatial mean-pool).

    FERAL aggregates V-JEPA2 patch tokens with learnable query tokens that
    cross-attend the full token set (MultiheadAttention) instead of averaging —
    letting the model focus on the mice / interaction region rather than diluting
    it with cage background. We apply the same per token-frame so the temporal
    axis (Tt frames/window) is preserved for the pyramid + DFC head.

    Residual form ``out = mean + gamma * attn(query, tokens)`` with ``gamma`` init
    0.1: the output starts ~= spatial mean-pool (the validated 91.71 mAP starting
    point) and the attention learns a correction on top — a strict superset of
    mean-pool that cannot regress below it at init.
    """

    def __init__(self, embed_dims, num_heads=16, num_queries=1, gamma_init=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, embed_dims) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.q_norm = nn.LayerNorm(embed_dims)
        self.kv_norm = nn.LayerNorm(embed_dims)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, g):
        # g: [BW, Tt, S, C] -> [BW, Tt, C]
        BW, Tt, S, C = g.shape
        mean = g.mean(dim=2)                                  # [BW, Tt, C]
        tok = self.kv_norm(g.reshape(BW * Tt, S, C))          # [BW*Tt, S, C]
        q = self.q_norm(self.query).unsqueeze(0).expand(BW * Tt, -1, -1)  # [BW*Tt, nq, C]
        out, _ = self.attn(q, tok, tok, need_weights=False)   # [BW*Tt, nq, C]
        out = out.mean(dim=1).reshape(BW, Tt, C)              # [BW, Tt, C]
        return mean + self.gamma * out


class FeralAttentionPool(nn.Module):
    """FERAL's AttentionPoolingBlockCustom (feral/model.py) as a spatiotemporal pooler.

    ``out_tokens`` learnable queries cross-attend the FULL window token set
    (Tt*S spatiotemporal tokens) -> ``out_tokens`` pooled vectors. Unlike per-frame
    spatial pooling, the queries attend across TIME, so each output token-frame can
    integrate the ordered temporal context of the whole window — the lever for the
    "investigation = ordered sequence" class (FERAL inv 93.3 vs our ~89). Faithful to
    FERAL: xavier queries, LayerNorm on q and kv, MultiheadAttention(16 heads), no
    residual. out_tokens == Tt so the output [BW, Tt, C] is a drop-in for mean-pool.
    """

    def __init__(self, embed_dim, out_tokens, num_heads=16):
        super().__init__()
        self.out_tokens = out_tokens
        self.x_q = nn.Parameter(torch.empty(out_tokens, embed_dim))
        nn.init.xavier_uniform_(self.x_q)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_x = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, g):
        # g: [BW, Tt, S, C] -> [BW, out_tokens(=Tt), C]
        BW, Tt, S, C = g.shape
        x = self.ln_x(g.reshape(BW, Tt * S, C))               # full window tokens
        q = self.ln_q(self.x_q).unsqueeze(0).expand(BW, -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)        # [BW, out_tokens, C]
        return out


class VJEPA2BlockAdapter(nn.Module):
    """Bottleneck adapter inserted after each V-JEPA2 block (encoder FROZEN).

    Mirrors TRACE's ViT-S ``Adapter`` (vit_adapter.py): down -> GELU -> temporal
    depthwise conv (+1x1) -> up, with ``up`` zero-init and a learnable ``gamma`` so
    it starts as identity (== the frozen-backbone forward) and learns a residual.
    The depthwise conv runs over the Tt token-frames within each window => it injects
    LOCAL TEMPORAL modeling inside the otherwise-frozen encoder — the FERAL-analysis
    lever — at a fraction of the params of a top-12-block unfreeze (PEFT).

    tokens x: [BW, Tt*S, C] (Tt token-frames outer, S=g*g spatial inner).
    """

    def __init__(self, embed_dims, tt, g, mlp_ratio=0.25, kernel_size=3):
        super().__init__()
        hidden = max(8, int(embed_dims * mlp_ratio))
        self.tt, self.g = tt, g
        self.down = nn.Linear(embed_dims, hidden)
        self.act = nn.GELU()
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden)
        self.pw = nn.Conv1d(hidden, hidden, 1)
        self.up = nn.Linear(hidden, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        nn.init.trunc_normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        inp = x
        h = self.act(self.down(x))                       # [BW, N, hid]
        BW, N, hid = h.shape
        S = self.g * self.g
        # temporal depthwise conv over Tt at each spatial location
        z = h.reshape(BW, self.tt, S, hid).permute(0, 2, 3, 1).reshape(BW * S, hid, self.tt)
        z = self.pw(self.dwconv(z))
        z = z.reshape(BW, S, hid, self.tt).permute(0, 3, 1, 2).reshape(BW, N, hid)
        h = h + z
        return inp + self.gamma * self.up(h)


class VJEPA2Backbone(nn.Module):
    """Differentiable V-JEPA 2 ViT-L encoder wrapped as a TRACE backbone.

    Forward signature matches BackboneWrapper: forward(frames, masks=None) where
    `frames` is [B, num_segs, C, T, H, W] already pixel-normalized by the
    BackboneWrapper-style mean/std applied in this module's forward (we keep the
    normalization here so the dataset pipeline stays pixel/uint8-like and the
    config carries ImageNet mean/std on the 0-255 scale, mirroring the K400 path).

    Output: [B, embed_dims, T] feature map (channel-first), T == total_frames.
    """

    def __init__(
        self,
        model_id="facebook/vjepa2-vitl-fpc32-256-diving48",
        crop=256,
        fpc=32,            # frames fed to the encoder per window
        tubelet=2,         # -> fpc/tubelet token-frames per window
        patch=16,          # -> (crop/patch)^2 spatial tokens
        total_frames=768,  # clip length T; must be divisible by fpc
        embed_dims=1024,
        mean=(123.675, 116.28, 103.53),   # ImageNet mean on 0-255 scale
        std=(58.395, 57.12, 57.375),      # ImageNet std on 0-255 scale
        gradient_checkpointing=True,
        freeze_first_n_blocks=0,          # freeze the lowest-N transformer blocks
        freeze_backbone=False,            # full freeze (probe/frozen mode)
        norm_eval=False,
        local_files_only=True,
        spatial_pool="mean",              # "mean" (legacy) or "attn" (FERAL-style)
        attn_pool_heads=16,
        attn_pool_queries=1,
        attn_pool_gamma_init=0.1,
        adapter=None,                     # None | "conv": PEFT — freeze encoder, train per-block adapters
        adapter_mlp_ratio=0.25,
    ):
        super().__init__()
        from transformers import VJEPA2Model

        self.crop = crop
        self.fpc = fpc
        self.tubelet = tubelet
        self.patch = patch
        self.total_frames = total_frames
        self.embed_dims = embed_dims
        self.spatial_tokens = (crop // patch) ** 2          # 256
        self.token_frames_per_window = fpc // tubelet       # 16
        assert total_frames % fpc == 0, (
            f"total_frames ({total_frames}) must be divisible by fpc ({fpc})"
        )
        self.num_windows = total_frames // fpc              # 24
        self.freeze_backbone = freeze_backbone
        self.norm_eval = norm_eval

        # --- build the V-JEPA 2 model, keep ONLY the encoder ---------------
        full = VJEPA2Model.from_pretrained(
            model_id, local_files_only=local_files_only, torch_dtype=torch.float32
        )
        # Drop the JEPA predictor + (already-unloaded) pooler/classifier so they
        # never enter the optimizer / consume memory.
        self.encoder = full.encoder
        del full

        # --- normalization buffers (BackboneWrapper-style, 0-255 scale) -----
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1, 1)
        )

        # --- gradient checkpointing on the 24 transformer blocks ------------
        # This transformers version's VJEPA2Encoder.forward does NOT have a
        # gradient-checkpointing branch (it just loops self.layer). So we run a
        # custom forward (_encode_windows) that checkpoints each VJEPA2Layer with
        # use_reentrant=False — avoids the fork_rng path that crashed the K400 run.
        self.gradient_checkpointing = gradient_checkpointing and not freeze_backbone

        # --- freezing -------------------------------------------------------
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif freeze_first_n_blocks > 0:
            # encoder.layer is a ModuleList of 24 VJEPA2Layer blocks.
            for i, blk in enumerate(self.encoder.layer):
                if i < freeze_first_n_blocks:
                    for p in blk.parameters():
                        p.requires_grad = False
            # also freeze the patch embeddings with the lowest blocks
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False

        # --- spatial pooling head (mean-pool vs FERAL-style attention pool) ---
        self.spatial_pool = str(spatial_pool).lower()
        self.attn_pool = None
        if self.spatial_pool == "attn":
            self.attn_pool = SpatialAttentionPool(
                embed_dims, num_heads=attn_pool_heads,
                num_queries=attn_pool_queries, gamma_init=attn_pool_gamma_init,
            )
        elif self.spatial_pool == "feral":
            # FERAL cross-time attention pooling: out_tokens=Tt queries attend the
            # full window's spatiotemporal tokens (drop-in for mean-pool).
            self.attn_pool = FeralAttentionPool(
                embed_dims, out_tokens=self.token_frames_per_window, num_heads=attn_pool_heads,
            )
        elif self.spatial_pool != "mean":
            raise ValueError(f"Unsupported spatial_pool: {spatial_pool!r}")

        # --- PEFT adapters: freeze the ENTIRE encoder (overrides freeze_first_n) and
        # train a per-block bottleneck adapter instead. Far fewer trainable params than
        # a top-12-block unfreeze -> less overfit on a small train set; the adapter's
        # temporal dwconv still injects local temporal modeling. ---
        self.adapters = None
        if adapter is not None:
            if str(adapter).lower() != "conv":
                raise ValueError(f"Unsupported adapter: {adapter!r}")
            for p in self.encoder.parameters():
                p.requires_grad = False
            g = int(round(self.spatial_tokens ** 0.5))
            self.adapters = nn.ModuleList([
                VJEPA2BlockAdapter(embed_dims, self.token_frames_per_window, g,
                                   mlp_ratio=adapter_mlp_ratio)
                for _ in range(len(self.encoder.layer))
            ])

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"[VJEPA2Backbone] model={model_id} crop={crop} fpc={fpc} "
            f"windows={self.num_windows} token_frames/window={self.token_frames_per_window} "
            f"trainable={n_train/1e6:.1f}M / total={n_total/1e6:.1f}M "
            f"grad_ckpt={gradient_checkpointing} freeze_backbone={freeze_backbone} "
            f"freeze_first_n={freeze_first_n_blocks}"
        )

    # ------------------------------------------------------------------
    def _run_encoder(self, x):
        """Custom V-JEPA2 encoder forward with optional per-layer gradient
        checkpointing. x: [BW, fpc, 3, crop, crop] -> last_hidden_state [BW, N, C].
        Mirrors VJEPA2Encoder.forward (embeddings -> 24 layers -> layernorm) but
        checkpoints each VJEPA2Layer (use_reentrant=False) when training."""
        hidden_states = self.encoder.embeddings(x)
        use_ckpt = self.gradient_checkpointing and self.training
        for i, layer_module in enumerate(self.encoder.layer):
            if use_ckpt:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer_module, hidden_states, None, use_reentrant=False
                )
            else:
                layer_outputs = layer_module(hidden_states, None)
            hidden_states = layer_outputs[0]
            if self.adapters is not None:
                hidden_states = self.adapters[i](hidden_states)
        hidden_states = self.encoder.layernorm(hidden_states)
        return hidden_states

    def _encode_windows(self, x):
        """x: [BW, fpc, 3, crop, crop] -> [BW, token_frames, embed_dims]
        (spatial mean-pooled per token-frame, differentiable)."""
        lhs = self._run_encoder(x)                      # [BW, Tt*S, C]
        BW, N, C = lhs.shape
        S = self.spatial_tokens
        Tt = N // S
        g = lhs.reshape(BW, Tt, S, C)                   # [BW, Tt, S, C]
        if self.attn_pool is not None:
            pooled = self.attn_pool(g)                  # FERAL-style attn pool -> [BW, Tt, C]
        else:
            pooled = g.mean(dim=2)                      # spatial mean-pool -> [BW, Tt, C]
        return pooled

    def forward(self, frames, masks=None):
        """frames: [B, num_segs, C, T, H, W] (pixel scale, e.g. 0..255 floats).
        Returns [B, embed_dims, T] (channel-first), T == total_frames."""
        if self.norm_eval:
            self.set_norm_eval()

        B, num_segs = frames.shape[0], frames.shape[1]
        assert num_segs == 1, f"VJEPA2Backbone expects num_segs=1, got {num_segs}"
        x = frames[:, 0]                                # [B, C, T, H, W]
        C, T, H, W = x.shape[1:]
        assert T == self.total_frames, (
            f"expected T={self.total_frames}, got {T}"
        )

        # normalize (ImageNet mean/std on the incoming pixel scale)
        x = (x.float() - self.mean) / self.std          # [B, C, T, H, W]

        # spatial resize to the V-JEPA 2 crop if needed
        if H != self.crop or W != self.crop:
            x = x.reshape(B * C, T, H, W)
            x = F.interpolate(x, size=(self.crop, self.crop), mode="bilinear",
                              align_corners=False)
            x = x.reshape(B, C, T, self.crop, self.crop)
            H = W = self.crop

        # tile into contiguous fpc-frame windows:
        # [B, C, T, H, W] -> [B, C, num_windows, fpc, H, W] -> [B*num_windows, fpc, C, H, W]
        nw, fpc = self.num_windows, self.fpc
        x = x.reshape(B, C, nw, fpc, H, W)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()    # [B, nw, fpc, C, H, W]
        x = x.reshape(B * nw, fpc, C, H, W)             # [B*nw, fpc, C, H, W]

        # encode all windows of all clips in one batch (gradient checkpointed)
        if self.freeze_backbone:
            with torch.no_grad():
                pooled = self._encode_windows(x)        # [B*nw, Tt, embed]
        else:
            pooled = self._encode_windows(x)

        Tt = self.token_frames_per_window
        embed = pooled.shape[-1]
        # [B*nw, Tt, embed] -> [B, nw*Tt, embed] -> [B, embed, nw*Tt]
        pooled = pooled.reshape(B, nw * Tt, embed)
        pooled = pooled.permute(0, 2, 1).contiguous()   # [B, embed, nw*Tt]

        # differentiable temporal upsample nw*Tt (=384) -> total_frames (=768)
        feats = F.interpolate(pooled.float(), size=self.total_frames,
                              mode="linear", align_corners=False)  # [B, embed, T]

        if masks is not None and feats.dim() == 3:
            feats = feats * masks.unsqueeze(1).detach().float()

        return feats.to(torch.float32)

    # ------------------------------------------------------------------
    def set_norm_eval(self):
        for m in self.modules():
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.eval()
