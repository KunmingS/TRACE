# Pure-Python 1D NMS for temporal action detection.
# Vectorized PyTorch implementation — zero build dependencies.
import torch


def nms_1d(segs, scores, iou_threshold):
    """Greedy 1D NMS with vectorized suppression. Returns indices sorted by descending score."""
    if segs.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    x1 = segs[:, 0]
    x2 = segs[:, 1]
    areas = x2 - x1 + 1e-6

    order = scores.sort(descending=True).indices
    # Reorder all arrays by descending score
    x1 = x1[order]
    x2 = x2[order]
    areas = areas[order]

    keep = []
    alive = torch.ones(len(order), dtype=torch.bool)

    for i in range(len(order)):
        if not alive[i]:
            continue
        keep.append(order[i])

        # Vectorized IoU against all remaining alive segments
        remaining = alive.clone()
        remaining[: i + 1] = False
        if not remaining.any():
            break

        inter = (torch.min(x2[i], x2[remaining]) - torch.max(x1[i], x1[remaining])).clamp(min=0)
        ovr = inter / (areas[i] + areas[remaining] - inter)
        # Suppress overlapping segments
        suppress_mask = ovr >= iou_threshold
        # Map back to full indices
        remaining_indices = remaining.nonzero(as_tuple=True)[0]
        alive[remaining_indices[suppress_mask]] = False

    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long)


def softnms_1d(segs, scores, iou_threshold, sigma, min_score, method, t1, t2):
    """Soft-NMS with vectorized score decay. Returns (sorted_segs_with_scores [N,3], kept_indices)."""
    if segs.numel() == 0:
        return torch.empty((0, 3)), torch.empty(0, dtype=torch.long)

    x1 = segs[:, 0].clone()
    x2 = segs[:, 1].clone()
    sc = scores.clone()
    areas = x2 - x1 + 1e-6
    inds = torch.arange(len(sc), dtype=torch.long)
    n = len(sc)

    dets = torch.empty((n, 3))
    kept_inds = torch.empty(n, dtype=torch.long)
    num_kept = 0

    i = 0
    while i < n:
        # find max score from position i onward
        max_pos = i + sc[i:n].argmax()
        # swap i and max_pos
        if max_pos != i:
            for arr in (x1, x2, sc, areas, inds):
                arr[i], arr[max_pos] = arr[max_pos].clone(), arr[i].clone()

        ix1, ix2, iscore, iarea = x1[i], x2[i], sc[i], areas[i]
        dets[num_kept] = torch.stack([ix1, ix2, iscore])
        kept_inds[num_kept] = inds[i]
        num_kept += 1

        if i + 1 >= n:
            break

        i_next = i + 1
        # Vectorized IoU computation for all remaining segments
        rem_x1 = x1[i + 1 : n]
        rem_x2 = x2[i + 1 : n]
        rem_areas = areas[i + 1 : n]

        inter = (torch.min(ix2, rem_x2) - torch.max(ix1, rem_x1)).clamp(min=0)
        ovr = inter / (iarea + rem_areas - inter)

        # Vectorized weight computation
        if method == 0:  # vanilla
            weights = torch.where(ovr >= iou_threshold, torch.zeros_like(ovr), torch.ones_like(ovr))
        elif method == 1:  # linear
            weights = torch.where(ovr >= iou_threshold, 1.0 - ovr, torch.ones_like(ovr))
        elif method == 2:  # gaussian
            weights = torch.exp(-(ovr * ovr) / sigma)
        elif method == 3:  # improved gaussian (BMN)
            threshold = t1 + t2 * iarea
            weights = torch.where(ovr >= threshold, torch.exp(-(ovr * ovr) / sigma), torch.ones_like(ovr))
        else:
            weights = torch.ones_like(ovr)

        # Apply weights vectorized
        sc[i + 1 : n] *= weights

        # Remove segments below min_score (compact the arrays)
        valid = sc[i + 1 : n] >= min_score
        num_valid = valid.sum().item()
        if num_valid < n - i - 1:
            # Compact: move valid elements to front
            valid_idx = valid.nonzero(as_tuple=True)[0] + i + 1
            new_n = i + 1 + num_valid
            for arr in (x1, x2, sc, areas, inds):
                arr[i + 1 : new_n] = arr[valid_idx].clone()
            n = new_n

        i += 1  # must be at while-loop level

    dets_t = dets[:num_kept]
    inds_t = kept_inds[:num_kept]
    return dets_t, inds_t


class NMSop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, segs, scores, cls_idxs, iou_threshold, min_score, max_num):
        is_filtering_by_score = min_score > 0
        if is_filtering_by_score:
            valid_mask = scores > min_score
            segs, scores = segs[valid_mask], scores[valid_mask]
            cls_idxs = cls_idxs[valid_mask]

        inds = nms_1d(segs.cpu(), scores.cpu(), iou_threshold=float(iou_threshold))

        if max_num > 0:
            inds = inds[: min(max_num, len(inds))]
        return segs[inds].clone(), scores[inds].clone(), cls_idxs[inds].clone()


class SoftNMSop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, segs, scores, cls_idxs, iou_threshold, sigma, min_score, method, max_num, t1, t2):
        dets, inds = softnms_1d(
            segs.cpu(), scores.cpu(),
            iou_threshold=float(iou_threshold),
            sigma=float(sigma),
            min_score=float(min_score),
            method=int(method),
            t1=float(t1),
            t2=float(t2),
        )

        n_segs = min(len(inds), max_num) if max_num > 0 else len(inds)
        sorted_segs = dets[:n_segs, :2]
        sorted_scores = dets[:n_segs, 2]
        sorted_cls_idxs = cls_idxs[inds[:n_segs]]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


def seg_voting(nms_segs, all_segs, all_scores, iou_threshold, score_offset=1.5):
    """Bounding box voting — refine localization using neighboring segments."""
    num_nms_segs, num_all_segs = nms_segs.shape[0], all_segs.shape[0]
    ex_nms_segs = nms_segs[:, None].expand(num_nms_segs, num_all_segs, 2)
    ex_all_segs = all_segs[None, :].expand(num_nms_segs, num_all_segs, 2)

    left = torch.maximum(ex_nms_segs[:, :, 0], ex_all_segs[:, :, 0])
    right = torch.minimum(ex_nms_segs[:, :, 1], ex_all_segs[:, :, 1])
    inter = (right - left).clamp(min=0)

    nms_seg_lens = ex_nms_segs[:, :, 1] - ex_nms_segs[:, :, 0]
    all_seg_lens = ex_all_segs[:, :, 1] - ex_all_segs[:, :, 0]
    iou = inter / (nms_seg_lens + all_seg_lens - inter)

    seg_weights = (iou >= iou_threshold).to(all_scores.dtype) * all_scores[None, :]
    seg_weights /= torch.sum(seg_weights, dim=1, keepdim=True)
    return seg_weights @ all_segs


def batched_nms(
    segs,
    scores,
    cls_idxs,
    iou_threshold=0.0,
    min_score=0.0,
    max_seg_num=100,
    use_soft_nms=True,
    multiclass=True,
    sigma=0.5,
    voting_thresh=0.0,
    method=2,
    t1=0,
    t2=0,
):
    segs = segs.float()
    scores = scores.float()

    num_segs = segs.shape[0]
    if num_segs == 0:
        return (
            torch.zeros([0, 2]),
            torch.zeros([0]),
            torch.zeros([0], dtype=cls_idxs.dtype),
        )

    if multiclass:
        new_segs, new_scores, new_cls_idxs = [], [], []
        for class_id in torch.unique(cls_idxs):
            curr_indices = torch.where(cls_idxs == class_id)[0]
            if use_soft_nms:
                sorted_segs, sorted_scores, sorted_cls_idxs = SoftNMSop.apply(
                    segs[curr_indices], scores[curr_indices], cls_idxs[curr_indices],
                    iou_threshold, sigma, min_score, method, max_seg_num, t1, t2,
                )
            else:
                sorted_segs, sorted_scores, sorted_cls_idxs = NMSop.apply(
                    segs[curr_indices], scores[curr_indices], cls_idxs[curr_indices],
                    iou_threshold, min_score, max_seg_num,
                )
            new_segs.append(sorted_segs)
            new_scores.append(sorted_scores)
            new_cls_idxs.append(sorted_cls_idxs)

        new_segs = torch.cat(new_segs)
        new_scores = torch.cat(new_scores)
        new_cls_idxs = torch.cat(new_cls_idxs)
    else:
        if use_soft_nms:
            new_segs, new_scores, new_cls_idxs = SoftNMSop.apply(
                segs, scores, cls_idxs,
                iou_threshold, sigma, min_score, method, max_seg_num, t1, t2,
            )
        else:
            new_segs, new_scores, new_cls_idxs = NMSop.apply(
                segs, scores, cls_idxs, iou_threshold, min_score, max_seg_num,
            )
        if voting_thresh > 0:
            new_segs = seg_voting(new_segs, segs, scores, voting_thresh)

    _, idxs = new_scores.sort(descending=True)
    max_seg_num = min(max_seg_num, new_segs.shape[0])
    new_segs = new_segs[idxs[:max_seg_num]]
    new_scores = new_scores[idxs[:max_seg_num]]
    new_cls_idxs = new_cls_idxs[idxs[:max_seg_num]]
    return new_segs, new_scores, new_cls_idxs
