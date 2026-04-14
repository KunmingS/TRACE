import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import NECKS


class DeformableSample1D(nn.Module):
    """1D deformable sampling using grid_sample.

    Given input features and learned temporal offsets, performs deformable
    resampling along the temporal dimension.

    Args:
        channels: Number of input channels (used for offset prediction).
    """

    def __init__(self, channels):
        super().__init__()
        # Predict per-position temporal offset from concatenated features
        self.offset_conv = nn.Conv1d(channels * 2, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, source, target, mask=None):
        """
        Args:
            source: Feature to be resampled [B, C, T_src].
            target: Reference feature at target resolution [B, C, T_tgt].
            mask: Optional mask [B, 1, T_tgt].

        Returns:
            Deformably sampled features [B, C, T_tgt].
        """
        B, C, T_tgt = target.shape
        T_src = source.shape[2]

        # Interpolate source to target resolution first
        if T_src != T_tgt:
            source_interp = F.interpolate(source, size=T_tgt, mode="linear", align_corners=False)
        else:
            source_interp = source

        # Predict offsets from concatenated features
        combined = torch.cat([target, source_interp], dim=1)  # [B, 2C, T_tgt]
        offsets = self.offset_conv(combined)  # [B, 1, T_tgt]

        # Build sampling grid: normalized positions + offsets
        # Grid positions in [-1, 1] for grid_sample
        base_grid = torch.linspace(-1, 1, T_tgt, device=target.device).view(1, 1, T_tgt).expand(B, -1, -1)
        # Scale offsets relative to temporal extent
        # offsets are in units of positions, normalize to [-1, 1] range
        offset_scale = 2.0 / max(T_src - 1, 1)
        sample_grid = base_grid + offsets * offset_scale
        sample_grid = sample_grid.clamp(-1, 1)

        # Reshape for grid_sample: [B, C, 1, T_src] -> sample with [B, 1, T_tgt, 2]
        source_4d = source.unsqueeze(2)  # [B, C, 1, T_src]
        grid_2d = torch.zeros(B, 1, T_tgt, 2, device=target.device)
        grid_2d[:, 0, :, 0] = sample_grid[:, 0, :]  # temporal dimension [B, T_tgt]
        # grid_2d[..., 1] stays 0 (dummy spatial dimension)

        sampled = F.grid_sample(source_4d, grid_2d, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.squeeze(2)  # [B, C, T_tgt]

        if mask is not None:
            sampled = sampled * mask

        return sampled


@NECKS.register_module()
class TemporalDeformableFPN(nn.Module):
    """Bidirectional temporal FPN with deformable cross-scale fusion.

    Performs top-down and bottom-up feature fusion across FPN levels using
    learned temporal offsets and gated aggregation.

    Args:
        in_channels: Number of input/output channels (must be same for all levels).
        num_levels: Number of FPN levels.
        norm_cfg: Normalization config (default: LayerNorm).
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        num_levels=6,
        norm_cfg=dict(type="LN"),
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels
        assert in_channels == out_channels, "TemporalDeformableFPN requires in_channels == out_channels"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels

        # Top-down pathway: deformable sampling + gating
        self.td_deform = nn.ModuleList()
        self.td_gate = nn.ModuleList()
        for i in range(num_levels - 1):
            self.td_deform.append(DeformableSample1D(in_channels))
            self.td_gate.append(nn.Sequential(
                nn.Conv1d(in_channels * 2, in_channels, kernel_size=1),
                nn.Sigmoid(),
            ))

        # Bottom-up pathway: strided conv + deformable sampling + gating
        self.bu_downsample = nn.ModuleList()
        self.bu_deform = nn.ModuleList()
        self.bu_gate = nn.ModuleList()
        for i in range(num_levels - 1):
            self.bu_downsample.append(nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1))
            self.bu_deform.append(DeformableSample1D(in_channels))
            self.bu_gate.append(nn.Sequential(
                nn.Conv1d(in_channels * 2, in_channels, kernel_size=1),
                nn.Sigmoid(),
            ))

        # Output projection with residual
        self.out_convs = nn.ModuleList()
        self.out_norms = nn.ModuleList()
        for i in range(num_levels):
            self.out_convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            self.out_norms.append(nn.LayerNorm(out_channels, eps=1e-6))

    def forward(self, inputs, fpn_masks):
        """
        Args:
            inputs: Tuple of features per level, each [B, C, T_l].
            fpn_masks: Tuple of masks per level, each [B, T_l].

        Returns:
            Tuple of (fpn_feats, fpn_masks).
        """
        assert len(inputs) == self.num_levels
        assert len(fpn_masks) == self.num_levels

        # Copy to list for in-place updates
        feats = list(inputs)

        # Top-down pathway (from coarsest to finest)
        for l in range(self.num_levels - 2, -1, -1):
            # feats[l+1] is coarser, feats[l] is finer
            mask_l = fpn_masks[l].unsqueeze(1).float()  # [B, 1, T_l]

            # Deformable sampling of coarser features at finer resolution
            f_deform = self.td_deform[l](feats[l + 1], feats[l], mask_l)

            # Gated aggregation
            gate = self.td_gate[l](torch.cat([feats[l], f_deform], dim=1))
            feats[l] = feats[l] + gate * f_deform

        # Bottom-up pathway (from finest to coarsest)
        for l in range(self.num_levels - 1):
            # feats[l] is finer, feats[l+1] is coarser
            mask_lp1 = fpn_masks[l + 1].unsqueeze(1).float()  # [B, 1, T_{l+1}]

            # Downsample finer features
            f_down = self.bu_downsample[l](feats[l])

            # Deformable sampling
            f_deform = self.bu_deform[l](f_down, feats[l + 1], mask_lp1)

            # Gated aggregation
            gate = self.bu_gate[l](torch.cat([feats[l + 1], f_deform], dim=1))
            feats[l + 1] = feats[l + 1] + gate * f_deform

        # Output projection + residual + norm
        fpn_feats = tuple()
        for l in range(self.num_levels):
            out = self.out_convs[l](feats[l]) + feats[l]
            # LN requires [B, T, C] format
            out = self.out_norms[l](out.permute(0, 2, 1)).permute(0, 2, 1)
            fpn_feats += (out,)

        return fpn_feats, fpn_masks
