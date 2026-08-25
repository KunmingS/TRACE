from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
import copy

from .bricks import ConvModule
from .builder import NECKS


@NECKS.register_module()
class FPN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_levels,
        norm_cfg=None,
    ):
        super().__init__()

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        if isinstance(in_channels, int):
            in_channels = [in_channels] * num_levels

        for i in range(num_levels):
            self.lateral_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                )
            )
            self.fpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                )
            )

    def forward(self, input_list, mask_list):
        assert len(input_list) == len(self.lateral_convs)

        # build laterals
        laterals = [self.lateral_convs[i](input_list[i], mask_list[i])[0] for i in range(len(self.lateral_convs))]

        # build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode="nearest")

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i], mask_list[i])[0] for i in range(len(laterals))]
        return fpn_outs, mask_list


@NECKS.register_module()
class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,  # input feature channels, len(in_channels) = #levels
        out_channels,  # output feature channel
        num_levels=0,
        scale_factor=2.0,  # downsampling rate between two fpn levels
        start_level=0,  # start fpn level
        end_level=-1,  # end fpn level
        norm_cfg=dict(type="LN"),  # if no norm, set to none
    ):
        super().__init__()

        self.in_channels = [in_channels] * num_levels
        self.out_channel = out_channels
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(self.in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(self.in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        if norm_cfg is not None:
            norm_cfg = copy.copy(norm_cfg)  # make a copy
            norm_type = norm_cfg["type"]
            norm_cfg.pop("type")
            self.norm_type = norm_type
        else:
            self.norm_type = None

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i] == self.out_channel

            if self.norm_type == "BN":
                fpn_norm = nn.BatchNorm1d(num_features=out_channels, **norm_cfg)
            elif self.norm_type == "GN":
                fpn_norm = nn.GroupNorm(num_channels=out_channels, **norm_cfg)
            elif self.norm_type == "LN":
                fpn_norm = nn.LayerNorm(out_channels, eps=1e-6)
            else:
                assert self.norm_type is None
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) == len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(len(self.fpn_norms)):
            x = inputs[i + self.start_level]
            if self.norm_type == "LN":
                x = self.fpn_norms[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.fpn_norms[i](x)
            fpn_feats += (x,)
            new_fpn_masks += (fpn_masks[i + self.start_level],)

        return fpn_feats, new_fpn_masks


# ── merged from spatial_attn_pool.py ──
"""FERAL-style learnable-query attention pooling over the spatial patch grid.

Replaces the backbone's spatial MEAN-pool: instead of averaging the G*G patch tokens
of each frame into one vector, a small set of learnable query tokens cross-attend over
the patch tokens (attentive pooling, à la FERAL's attention-pool head), so the spatial
aggregation is content-adaptive. Sits on the backbone feature MAP path (return_feat_map
=True), parallel to SpatialFusion, and outputs [B, C, window_size] like that path does
(temporal token axis interpolated to window_size, matching the mean-pool post-proc).

This is ORTHOGONAL to the time-axis DFC heads (conv/tcn/attn/asformer/ssm/dyfadet/routed):
it changes WHERE/how the spatial dim is pooled, they model the time dim afterwards.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttnPool(nn.Module):
    def __init__(self, in_channels, window_size, num_queries=4, num_heads=8, fuse_mean=False):
        super().__init__()
        self.window_size = int(window_size)
        self.nh = int(num_heads)
        # fuse_mean: concat mean-pool ⊕ SAP -> [B, 2C, window]. The downstream projection
        # then sees BOTH spatial views and learns a per-behavior mix (the "routed SAP"
        # idea as a learned fusion). projection in_channels must be 2*in_channels.
        self.fuse_mean = bool(fuse_mean)
        self.query = nn.Parameter(torch.randn(int(num_queries), in_channels) * 0.02)
        self.norm = nn.LayerNorm(in_channels)
        self.k = nn.Linear(in_channels, in_channels)
        self.v = nn.Linear(in_channels, in_channels)
        self.qp = nn.Linear(in_channels, in_channels)
        self.out = nn.Linear(in_channels, in_channels)

    def forward(self, feat_map):  # [B, C, T, G, G] -> [B, C (or 2C if fuse_mean), window_size]
        B, C, T, G, Gw = feat_map.shape
        x = feat_map.permute(0, 2, 3, 4, 1).reshape(B * T, G * Gw, C)   # [BT, P, C]
        x = self.norm(x)
        k, v = self.k(x), self.v(x)
        q = self.qp(self.query).unsqueeze(0).expand(B * T, -1, -1)       # [BT, Q, C]
        h, hd = self.nh, C // self.nh
        sp = lambda t: t.view(t.shape[0], t.shape[1], h, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(sp(q), sp(k), sp(v))         # [BT, h, Q, hd]
        o = o.transpose(1, 2).reshape(B * T, q.shape[1], C)
        o = self.out(o).mean(dim=1)                                    # [BT, C] avg over queries
        x_sap = o.reshape(B, T, C).transpose(1, 2)                     # [B, C, T]
        x_sap = F.interpolate(x_sap, size=self.window_size, mode="linear", align_corners=False)
        if not self.fuse_mean:
            return x_sap
        x_mean = feat_map.mean(dim=(3, 4))                             # [B, C, T] spatial mean-pool
        x_mean = F.interpolate(x_mean, size=self.window_size, mode="linear", align_corners=False)
        return torch.cat([x_mean, x_sap], dim=1)                       # [B, 2C, window]


# ── merged from spatial_fusion.py ──
"""Detector-side spatial fusion of the video / detail / frame-context branches.

This is the §4.1 "bbox-aligned conv fusion" of the scale-split plan, lifted OUT of
the backbone's ``post_processing_pipeline`` (where ``ConvSpatialPool`` could only see
the single ViT branch) into the detector, so the H/W grid of every branch survives
to be fused BEFORE pooling:

    sources = [ (V) crop-SSL ViT map, (D) detail-CNN map, (F) ROIAlign'd scene map,
                coord maps ]   each [B, C_i, T, G, G]
      -> resize each to the common grid G
      -> concat on channels
      -> 1x1 channel mix -> depthwise-separable 3x3 spatial conv blocks
      -> gated (or mean) spatial pool over (G, G)        -> [B, C_out, T_tok]
      -> temporal interpolate to window_size             -> [B, C_out, T]

Pooling each branch independently and concatenating (the old dual-backbone failure
mode, +0.0048) is exactly what this avoids: the conv sees all branches on one grid,
so e.g. the water-bottle's position relative to the mouse is visible to a 3x3 kernel.

Stage 1 wires only the (V) source and is designed to reproduce ``ConvSpatialPool``'s
behaviour (gated pool + mean-pool residual); (D)/(F)/coord are added as extra
sources without touching the pooling/​temporal path.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm: str | None, channels: int) -> nn.Module:
    norm = (norm or "none").lower()
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "gn":
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unsupported SpatialFusion norm: {norm!r}")


class SpatialFusion(nn.Module):
    """Fuse a list of ``[B, C_i, T, G, G]`` spatial branches into ``[B, C_out, T]``.

    Args:
        in_channels: channels of the primary (V) branch — also the residual width.
        source_channels: per-source channel counts in fusion order, e.g.
            ``[384]`` for (V)-only, ``[384, 64, 96, 4]`` for V+D+F+coord. The first
            entry MUST equal ``in_channels`` (the residual / primary branch).
        out_channels: output channel count (defaults to in_channels).
        grid_size: common spatial grid the branches are resized to (default: the
            primary branch's own grid, inferred at runtime if None).
        window_size: temporal length to interpolate the pooled feature to.
        hidden_channels / num_blocks / kernel_size / norm / pooling / residual /
        init_scale: as in ConvSpatialPool.
    """

    def __init__(
        self,
        in_channels: int,
        source_channels: list[int] | tuple[int, ...] | None = None,
        out_channels: int | None = None,
        grid_size: int | None = None,
        window_size: int = 768,
        hidden_channels: int = 128,
        num_blocks: int = 2,
        kernel_size: int = 3,
        norm: str | None = "BN",
        pooling: str = "gated",
        residual: bool = True,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.source_channels = list(source_channels) if source_channels else [self.in_channels]
        if self.source_channels[0] != self.in_channels:
            raise ValueError("source_channels[0] (the V/residual branch) must equal in_channels")
        self.out_channels = int(out_channels or in_channels)
        self.grid_size = grid_size
        self.window_size = int(window_size)
        self.hidden_channels = int(hidden_channels)
        self.pooling = pooling.lower()
        self.residual = bool(residual)

        total_in = int(sum(self.source_channels))
        pad = int(kernel_size) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(total_in, self.hidden_channels, kernel_size=1),
            _make_norm(norm, self.hidden_channels),
            nn.GELU(),
        ]
        for _ in range(int(num_blocks)):
            layers += [
                nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=kernel_size,
                          padding=pad, groups=self.hidden_channels),
                _make_norm(norm, self.hidden_channels),
                nn.GELU(),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1),
                _make_norm(norm, self.hidden_channels),
                nn.GELU(),
            ]
        layers.append(nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=1))
        self.body = nn.Sequential(*layers)

        if self.pooling == "gated":
            self.gate = nn.Conv2d(self.out_channels, 1, kernel_size=1)
        elif self.pooling == "mean":
            self.gate = None
        else:
            raise ValueError(f"Unsupported SpatialFusion pooling: {pooling!r}")

        if self.residual:
            if self.out_channels != self.in_channels:
                raise ValueError("residual=True requires out_channels == in_channels")
            self.scale = nn.Parameter(torch.full((1,), float(init_scale)))

        self.last_gate_entropy: float | None = None

    def _pool(self, y: torch.Tensor) -> torch.Tensor:
        # y: [BT, C_out, G, G] -> [BT, C_out]
        if self.pooling == "mean":
            return y.mean(dim=(-2, -1))
        logits = self.gate(y).flatten(-2)  # [BT, 1, G*G]
        weights = logits.softmax(dim=-1)
        with torch.no_grad():
            safe = weights.clamp_min(1e-12)
            self.last_gate_entropy = float((-(safe * safe.log()).sum(dim=-1).mean()).item())
        return (y.flatten(-2) * weights).sum(dim=-1)

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        """sources: list of [B, C_i, T, G_i, G_i] (channels match source_channels)."""
        if len(sources) != len(self.source_channels):
            raise ValueError(f"expected {len(self.source_channels)} sources, got {len(sources)}")
        primary = sources[0]
        b, _, t, g0, _ = primary.shape
        g = int(self.grid_size or g0)

        # reshape each to [B*T, C_i, g, g], temporally aligning to the primary's
        # T_tok and spatially resizing to the common grid
        flat = []
        for src, c in zip(sources, self.source_channels):
            if src.shape[2] != t:  # temporal token mismatch -> interpolate along T
                src = F.interpolate(src, size=(t, src.shape[3], src.shape[4]),
                                    mode="trilinear", align_corners=False)
            bb, cc, tt, gh, gw = src.shape
            x = src.permute(0, 2, 1, 3, 4).reshape(bb * tt, cc, gh, gw)
            if (gh, gw) != (g, g):
                x = F.interpolate(x, size=(g, g), mode="bilinear", align_corners=False)
            flat.append(x)
        x = torch.cat(flat, dim=1)  # [B*T, total_in, g, g]

        branch = self._pool(self.body(x))  # [B*T, out]
        branch = branch.view(b, t, self.out_channels).permute(0, 2, 1)  # [B, out, T_tok]

        if self.residual:
            legacy = primary.mean(dim=(-2, -1))  # [B, Cv, T_tok]
            out = legacy + self.scale * branch
        else:
            out = branch

        if out.shape[-1] != self.window_size:
            out = F.interpolate(out, size=self.window_size, mode="linear", align_corners=False)
        return out

    @property
    def residual_scale_value(self) -> float:
        return float(self.scale.detach().cpu().item()) if self.residual else 0.0

    def extra_repr(self) -> str:
        return (f"source_channels={self.source_channels}, out_channels={self.out_channels}, "
                f"grid_size={self.grid_size}, window_size={self.window_size}, pooling={self.pooling}")


class DetailCNN(nn.Module):
    """(D) detail branch — a light per-frame CNN over the crop for fine structures
    (tongue / paw / whiskers) that the ViT's 16px patchify destroys.

    Input crop frames ``[B, 3, T, H, W]`` (one stream). Frames are temporally
    subsampled by ``temporal_stride`` to match the ViT token rate, then a stride-2
    conv stack takes each frame down to the ``grid_size`` grid. Output
    ``[B, out_channels, T', g, g]`` is fed to ``SpatialFusion`` as an extra source.
    """

    def __init__(self, in_channels=3, out_channels=64, widths=(16, 32, 64),
                 temporal_stride=2, grid_size=14, norm="BN"):
        super().__init__()
        self.temporal_stride = int(temporal_stride)
        self.grid_size = int(grid_size)
        self.out_channels = int(out_channels)
        chans = [in_channels, *widths]
        blocks = []
        for ci, co in zip(chans[:-1], chans[1:]):
            blocks += [nn.Conv2d(ci, co, kernel_size=3, stride=2, padding=1),
                       _make_norm(norm, co), nn.GELU()]
        blocks.append(nn.Conv2d(chans[-1], out_channels, kernel_size=1))
        self.body = nn.Sequential(*blocks)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, 3, T, H, W]
        if self.temporal_stride > 1:
            frames = frames[:, :, :: self.temporal_stride]
        b, c, t, h, w = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
        x = self.body(x)  # [B*T, out, g', g']
        if x.shape[-1] != self.grid_size or x.shape[-2] != self.grid_size:
            x = F.interpolate(x, size=(self.grid_size, self.grid_size),
                              mode="bilinear", align_corners=False)
        return x.view(b, t, self.out_channels, self.grid_size, self.grid_size).permute(0, 2, 1, 3, 4)
