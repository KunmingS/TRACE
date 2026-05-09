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

    n = len(order)
    alive = torch.ones(n, dtype=torch.bool)
    keep = []

    for i in range(n):
        if not alive[i]:
            continue
        keep.append(order[i])

        if i + 1 >= n:
            break
        # Only check positions after i that are still alive
        tail_alive = alive[i + 1:]
        if not tail_alive.any():
            break

        # Vectorized IoU vs. the full tail, then mask by tail_alive.
        # Cheaper than nonzero+gather for typical N.
        inter = (torch.min(x2[i], x2[i + 1:]) - torch.max(x1[i], x1[i + 1:])).clamp(min=0)
        ovr = inter / (areas[i] + areas[i + 1:] - inter)
        suppress = (ovr >= iou_threshold) & tail_alive
        # In-place update through the slice view (PyTorch fancy-index assign).
        suppress_idx = suppress.nonzero(as_tuple=True)[0] + (i + 1)
        alive[suppress_idx] = False

    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long)


def softnms_1d(segs, scores, iou_threshold, sigma, min_score, method, t1, t2):
    """Soft-NMS with vectorized score decay. Returns (sorted_segs_with_scores [N,3], kept_indices).

    All five per-segment arrays (x1, x2, score, area, orig_idx) are packed into a single (n, 5)
    tensor so the hot-path row swap and compaction each become a single tensor op instead of
    looping over five parallel arrays in Python.
    """
    if segs.numel() == 0:
        return torch.empty((0, 3)), torch.empty(0, dtype=torch.long)

    n = segs.shape[0]
    x1 = segs[:, 0]
    x2 = segs[:, 1]
    areas = x2 - x1 + 1e-6
    packed = torch.stack(
        [x1, x2, scores, areas, torch.arange(n, dtype=segs.dtype)],
        dim=1,
    )

    i = 0
    cur_n = n
    while i < cur_n:
        # Move the max-scored remaining row to position i (one packed row swap).
        max_pos = i + int(packed[i:cur_n, 2].argmax())
        if max_pos != i:
            packed[[i, max_pos]] = packed[[max_pos, i]]

        ix1 = packed[i, 0]
        ix2 = packed[i, 1]
        iarea = packed[i, 3]

        if i + 1 < cur_n:
            # Vectorized IoU of kept segment vs. all remaining
            rem = packed[i + 1:cur_n]
            inter = (torch.min(ix2, rem[:, 1]) - torch.max(ix1, rem[:, 0])).clamp(min=0)
            ovr = inter / (iarea + rem[:, 3] - inter)

            if method == 0:  # vanilla (hard cutoff)
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

            packed[i + 1:cur_n, 2] *= weights

            # Compact: drop rows whose decayed score fell below min_score (one packed-slice op).
            valid = packed[i + 1:cur_n, 2] >= min_score
            num_valid = int(valid.sum())
            if num_valid < cur_n - i - 1:
                valid_idx = valid.nonzero(as_tuple=True)[0] + (i + 1)
                packed[i + 1:i + 1 + num_valid] = packed[valid_idx]
                cur_n = i + 1 + num_valid

        i += 1

    # Kept rows sit in packed[:i, :] in descending-score order by construction.
    kept_dets = packed[:i, :3].contiguous()
    kept_inds = packed[:i, 4].long()
    return kept_dets, kept_inds


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
