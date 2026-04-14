import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class BoundaryDistributionLoss(nn.Module):
    """KL-divergence loss between predicted boundary distributions and target
    Gaussian distributions centered at GT offsets.

    For each positive sample, constructs a Gaussian target distribution over
    the boundary bins, centered at the ground-truth offset. Computes
    KL(target || predicted) for both left and right boundaries.

    Args:
        tau: Standard deviation of the target Gaussian distribution.
        num_bins: Number of bins in the boundary distribution.
    """

    def __init__(self, tau=2.0, num_bins=16):
        super().__init__()
        self.tau = tau
        self.num_bins = num_bins

    def forward(self, pred_left_dis, pred_right_dis, gt_left, gt_right, reduction="sum"):
        """
        Args:
            pred_left_dis: Predicted left boundary distributions [N, num_bins+1],
                           already softmax-normalized.
            pred_right_dis: Predicted right boundary distributions [N, num_bins+1],
                            already softmax-normalized.
            gt_left: GT left offsets (normalized by stride) [N].
            gt_right: GT right offsets (normalized by stride) [N].
            reduction: 'none' | 'mean' | 'sum'.

        Returns:
            KL divergence loss.
        """
        if pred_left_dis.numel() == 0:
            return pred_left_dis.sum() * 0

        num_bins = self.num_bins
        device = pred_left_dis.device

        # Left boundary: bins go from num_bins down to 0 (reversed)
        left_bin_idx = torch.arange(num_bins, -1, -1, device=device, dtype=torch.float)  # [num_bins+1]
        # Right boundary: bins go from 0 to num_bins
        right_bin_idx = torch.arange(num_bins + 1, device=device, dtype=torch.float)  # [num_bins+1]

        # Build target Gaussian distributions
        # gt_left/gt_right are the expected values in bin-index space
        target_left = self._build_gaussian(left_bin_idx, gt_left, device)   # [N, num_bins+1]
        target_right = self._build_gaussian(right_bin_idx, gt_right, device)  # [N, num_bins+1]

        # KL(target || pred) = sum(target * log(target / pred))
        # Use F.kl_div which expects log(pred) as input
        log_pred_left = torch.log(pred_left_dis.clamp(min=1e-8))
        log_pred_right = torch.log(pred_right_dis.clamp(min=1e-8))

        kl_left = F.kl_div(log_pred_left, target_left, reduction="none").sum(dim=-1)
        kl_right = F.kl_div(log_pred_right, target_right, reduction="none").sum(dim=-1)

        loss = kl_left + kl_right

        if reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else loss.sum() * 0
        elif reduction == "sum":
            loss = loss.sum()
        return loss

    def _build_gaussian(self, bin_idx, gt_offset, device):
        """Build Gaussian target distribution centered at gt_offset.

        Args:
            bin_idx: Bin indices [num_bins+1].
            gt_offset: GT offsets [N].
            device: torch device.

        Returns:
            Normalized Gaussian distributions [N, num_bins+1].
        """
        # gt_offset: [N] -> [N, 1]
        gt_offset = gt_offset.unsqueeze(-1).clamp(min=0, max=self.num_bins)
        # bin_idx: [num_bins+1] -> [1, num_bins+1]
        bin_idx = bin_idx.unsqueeze(0)
        # Gaussian: exp(-(bin - gt)^2 / (2 * tau^2))
        logits = -((bin_idx - gt_offset) ** 2) / (2 * self.tau ** 2)
        # Normalize to form a proper distribution
        target = F.softmax(logits, dim=-1)
        return target


@LOSSES.register_module()
class BoundaryContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss pushing boundary-frame features apart from
    interior-frame features.

    Points within `boundary_ratio` of action start/end are positives.
    Points in the action interior are negatives.

    Args:
        temperature: Temperature for InfoNCE softmax.
        boundary_ratio: Fraction of action duration defining the boundary zone.
        max_negatives: Maximum number of negative samples to use per batch.
    """

    def __init__(self, temperature=0.07, boundary_ratio=0.1, max_negatives=256):
        super().__init__()
        self.temperature = temperature
        self.boundary_ratio = boundary_ratio
        self.max_negatives = max_negatives

    def forward(self, feat, points, gt_segments, gt_labels, pos_mask, valid_mask):
        """
        Args:
            feat: Concatenated features from all FPN levels [B, T, C].
            points: Concatenated point info [T, 4] (center, reg_min, reg_max, stride).
            gt_segments: List of [N_i, 2] GT segments per batch item.
            gt_labels: List of [N_i] GT labels per batch item.
            pos_mask: [B, T] boolean mask of positive (assigned) points.
            valid_mask: [B, T] boolean mask of valid points.

        Returns:
            InfoNCE loss scalar.
        """
        device = feat.device
        B, T, C = feat.shape
        total_loss = torch.tensor(0.0, device=device)
        num_valid = 0

        for b in range(B):
            if gt_segments[b].shape[0] == 0:
                continue

            pt_centers = points[:, 0]  # [T]
            boundary_mask = torch.zeros(T, device=device, dtype=torch.bool)
            interior_mask = torch.zeros(T, device=device, dtype=torch.bool)

            for seg in gt_segments[b]:
                seg_start, seg_end = seg[0], seg[1]
                seg_len = seg_end - seg_start
                boundary_zone = seg_len * self.boundary_ratio

                # Boundary: points near start or end
                near_start = (pt_centers >= seg_start - boundary_zone) & (pt_centers <= seg_start + boundary_zone)
                near_end = (pt_centers >= seg_end - boundary_zone) & (pt_centers <= seg_end + boundary_zone)
                boundary_mask |= near_start | near_end

                # Interior: points well inside the action (excluding boundary zones)
                in_interior = (pt_centers > seg_start + boundary_zone) & (pt_centers < seg_end - boundary_zone)
                interior_mask |= in_interior

            # Restrict to valid and positive points
            boundary_mask = boundary_mask & valid_mask[b]
            interior_mask = interior_mask & valid_mask[b] & ~boundary_mask

            n_boundary = boundary_mask.sum().item()
            n_interior = interior_mask.sum().item()

            if n_boundary < 2 or n_interior < 1:
                continue

            boundary_feats = F.normalize(feat[b, boundary_mask], dim=-1)  # [n_bnd, C]
            interior_feats = F.normalize(feat[b, interior_mask], dim=-1)  # [n_int, C]

            # Subsample negatives if too many
            if n_interior > self.max_negatives:
                perm = torch.randperm(n_interior, device=device)[:self.max_negatives]
                interior_feats = interior_feats[perm]

            # InfoNCE: for each boundary point, positives = other boundary points,
            # negatives = interior points
            # Similarity matrices
            pos_sim = torch.mm(boundary_feats, boundary_feats.t()) / self.temperature  # [n_bnd, n_bnd]
            neg_sim = torch.mm(boundary_feats, interior_feats.t()) / self.temperature  # [n_bnd, n_int]

            # Mask out self-similarity
            pos_mask_diag = ~torch.eye(n_boundary, device=device, dtype=torch.bool)
            pos_sim = pos_sim.masked_fill(~pos_mask_diag, float("-inf"))

            # For each anchor, logsumexp over all (positive + negative) as denominator
            # numerator is the positive similarities
            all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # [n_bnd, n_bnd + n_int]
            log_denom = torch.logsumexp(all_sim, dim=1)  # [n_bnd]

            # Average log-prob of positives
            # mask -inf entries (self) when computing mean
            pos_sim_valid = pos_sim.masked_fill(~pos_mask_diag, 0)
            n_pos_per_anchor = pos_mask_diag.sum(dim=1).float()  # n_bnd - 1
            sum_pos = pos_sim_valid.sum(dim=1)
            mean_pos = sum_pos / n_pos_per_anchor.clamp(min=1)

            loss_b = (-mean_pos + log_denom).mean()
            total_loss = total_loss + loss_b
            num_valid += 1

        if num_valid > 0:
            total_loss = total_loss / num_valid

        return total_loss
