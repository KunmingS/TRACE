import os
import pickle
import threading
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F


def boundary_choose(score):
    mask_high = score > score.max(dim=1, keepdim=True)[0] * 0.5
    mask_peak = score == F.max_pool1d(score, kernel_size=3, stride=1, padding=1)
    mask = mask_peak | mask_high
    return mask


# ── PTS table cache for `convert_to_seconds` ────────────────────────
#
# `convert_to_seconds` runs once per video at the end of the inference loop,
# in worker processes when DDP is on. Keep an LRU per-process cache of the
# clip-relative PTS slice so repeat lookups (e.g. across windows of the same
# clip) don't reload the .npy file. Modest cap because PTS arrays for a
# single 768-frame clip are only ~6 KB.
_PTS_CACHE_MAX = 64
_pts_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_pts_cache_lock = threading.Lock()


def _get_clip_pts(pts_path: str, source_frame_offset: int, clip_frame_count: int) -> np.ndarray:
    """Return clip-relative PTS slice (seconds, ``float64``).

    `pts_path` is the absolute path to the per-source-video PTS array
    cached as ``<video>.vtrace/pts.npy`` by ``data_prep._load_or_build_pts``.
    The returned slice is **clip-local**: the first PTS is rebased to 0.
    """
    key = (pts_path, int(source_frame_offset), int(clip_frame_count))
    with _pts_cache_lock:
        cached = _pts_cache.get(key)
        if cached is not None:
            _pts_cache.move_to_end(key)
            return cached

    full = np.load(pts_path).astype(np.float64, copy=False)
    end = int(source_frame_offset) + int(clip_frame_count)
    if end > len(full):
        end = len(full)
    sliced = full[int(source_frame_offset):end]
    sliced = sliced - sliced[0] if len(sliced) else sliced

    with _pts_cache_lock:
        _pts_cache[key] = sliced
        if len(_pts_cache) > _PTS_CACHE_MAX:
            _pts_cache.popitem(last=False)
    return sliced


def save_predictions(predictions, metas, folder):
    for idx in range(len(metas)):
        video_name = metas[idx]["video_name"]

        file_path = os.path.join(folder, f"{video_name}.pkl")
        prediction = [data[idx] for data in predictions]
        with open(file_path, "wb") as outfile:
            pickle.dump(prediction, outfile, pickle.HIGHEST_PROTOCOL)


def load_single_prediction(metas, folder):
    """Should not be used for sliding window. Since we saved the files with video name, and sliding window will have multiple files with the same name."""
    predictions = []
    for idx in range(len(metas)):
        video_name = metas[idx]["video_name"]
        file_path = os.path.join(folder, f"{video_name}.pkl")
        with open(file_path, "rb") as infile:
            prediction = pickle.load(infile)
        predictions.append(prediction)

    batched_predictions = []
    for i in range(len(predictions[0])):
        data = torch.stack([prediction[i] for prediction in predictions])
        batched_predictions.append(data)
    return batched_predictions


def load_predictions(metas, infer_cfg):
    if "fuse_list" in infer_cfg.keys():
        predictions = []
        predictions_list = [load_single_prediction(metas, folder) for folder in infer_cfg.fuse_list]
        for i in range(len(predictions_list[0])):
            predictions.append(torch.stack([pred[i] for pred in predictions_list]).mean(dim=0))
        return predictions
    else:
        return load_single_prediction(metas, infer_cfg.folder)


def convert_to_seconds(segments, meta):
    """Convert model-frame indices to clip-relative seconds.

    Two paths:

    1. **PTS-aware** (preferred when the dataset carries
       ``source_pts_table``): look up each model-frame index in the
       per-clip PTS table via piecewise-linear interpolation. Correct for
       both CFR and VFR sources — see
       ``pts-based-frame-mapping.md (archived)``.
    2. **Legacy CFR fallback**: `(idx * snippet_stride + …) / fps`.
       Identical to the historical formula when fps is constant; used for
       datasets prepped before the PTS upgrade.

    The two paths are bit-identical on a true CFR file: PTS values are
    exactly `i / fps` so the interpolation collapses to division.
    """
    if meta["fps"] == -1:  # resize setting, like in anet / hacs
        segments = segments / meta["resize_length"] * meta["duration"]
    else:  # sliding window / padding setting, like in thumos / ego4d
        snippet_stride = meta["snippet_stride"]
        offset_frames = meta["offset_frames"]
        window_start_frame = meta["window_start_frame"] if "window_start_frame" in meta.keys() else 0

        pts_path = meta.get("source_pts_table")
        clip_frame_count = meta.get("clip_frame_count")
        if pts_path and clip_frame_count:
            # PTS-aware mapping. `frame_idx` is in source-clip-local frame
            # coordinates; we interpolate against the clip-relative PTS
            # array to get clip-relative seconds.
            source_frame_offset = int(meta.get("source_frame_offset", 0))
            clip_pts = _get_clip_pts(pts_path, source_frame_offset, int(clip_frame_count))
            is_tensor = isinstance(segments, torch.Tensor)
            if is_tensor:
                seg_device = segments.device
                seg_dtype = segments.dtype
                seg_np = segments.detach().cpu().numpy().astype(np.float64, copy=False)
            else:
                seg_device = None
                seg_dtype = None
                seg_np = np.asarray(segments, dtype=np.float64)

            frame_idx = seg_np * snippet_stride + window_start_frame + offset_frames
            if len(clip_pts) > 0:
                xp = np.arange(len(clip_pts), dtype=np.float64)
                # `np.interp` clamps out-of-range to endpoints — same end
                # behaviour as the legacy `[0, duration]` truncation below.
                seconds_flat = np.interp(frame_idx.ravel(), xp, clip_pts)
                seconds = seconds_flat.reshape(frame_idx.shape)
            else:
                seconds = np.zeros_like(frame_idx)

            if is_tensor:
                segments = torch.from_numpy(seconds).to(device=seg_device, dtype=seg_dtype)
            else:
                segments = seconds
        else:
            # Legacy CFR fallback.
            segments = (segments * snippet_stride + window_start_frame + offset_frames) / meta["fps"]

    # truncate all boundaries within [0, duration]
    if segments.shape[0] > 0:
        segments[segments <= 0.0] *= 0.0
        segments[segments >= meta["duration"]] = segments[segments >= meta["duration"]] * 0.0 + meta["duration"]
    return segments


# ── merged from nms.py ──
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


# ── merged from classifier.py ──
import json
import numpy as np
import torch
from vtrace.registry import Registry

CLASSIFIERS = Registry("classifiers")


def build_classifier(cfg):
    """Build external classifier."""
    return CLASSIFIERS.build(cfg)


@CLASSIFIERS.register_module()
class CUHKANETClassifier:
    def __init__(self, path, topk=1):
        super().__init__()

        with open(path, "r") as f:
            cuhk_data = json.load(f)
        self.cuhk_data_score = cuhk_data["results"]
        self.cuhk_data_action = np.array(cuhk_data["class"])
        self.topk = topk

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)

        # sort video classification
        cuhk_score = np.array(self.cuhk_data_score[video_id])
        cuhk_classes = self.cuhk_data_action[np.argsort(-cuhk_score)]
        cuhk_score = cuhk_score[np.argsort(-cuhk_score)]

        new_segments = []
        new_labels = []
        new_scores = []
        # for segment, score in zip(segments, scores):
        for k in range(self.topk):
            new_segments.append(segments)
            new_labels.extend([cuhk_classes[k]] * len(segments))
            new_scores.append(scores * cuhk_score[k])

        new_segments = torch.cat(new_segments)
        new_scores = torch.cat(new_scores)
        return new_segments, new_labels, new_scores


@CLASSIFIERS.register_module()
class TCANetHACSClassifier:
    def __init__(self, path, topk=1):
        super().__init__()

        with open(path, "r") as f:
            cls_data = json.load(f)
        self.cls_data_score = cls_data["results"]
        self.cls_data_action = cls_data["class"]
        self.topk = topk

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)

        # sort video classification
        cls_score = np.array(self.cls_data_score[video_id][0])
        cls_score = np.exp(cls_score) / np.sum(np.exp(cls_score)) * 2.0
        cls_data_action = np.array(self.cls_data_action)
        cls_classes = cls_data_action[np.argsort(-cls_score)]
        cls_score = cls_score[np.argsort(-cls_score)]

        new_segments = []
        new_labels = []
        new_scores = []

        for k in range(self.topk):
            new_segments.append(segments)
            new_labels.extend([cls_classes[k]] * len(segments))
            new_scores.append(scores * cls_score[k])

        new_segments = torch.cat(new_segments)
        new_scores = torch.cat(new_scores)
        return new_segments, new_labels, new_scores


@CLASSIFIERS.register_module()
class StandardClassifier:
    def __init__(self, path, topk=1, apply_softmax=False):
        super().__init__()

        with open(path, "r") as f:
            cls_data = json.load(f)
        self.cls_data_score = cls_data["results"]
        self.cls_data_label = np.array(cls_data["class"]) if "class" in cls_data else np.array(cls_data["classes"])
        self.apply_softmax = apply_softmax
        self.topk = topk

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)
        cls_score = np.array(self.cls_data_score[video_id])

        if self.apply_softmax:  # do softmax
            cls_score = np.exp(cls_score) / np.sum(np.exp(cls_score))

        # sort video classification scores
        topk_cls_idx = np.argsort(cls_score)[::-1][: self.topk]
        topk_cls_score = cls_score[topk_cls_idx]
        topk_cls_label = self.cls_data_label[topk_cls_idx]

        new_segments = []
        new_labels = []
        new_scores = []

        for k in range(self.topk):
            new_segments.append(segments)
            new_labels.extend([topk_cls_label[k]] * len(segments))
            new_scores.append(np.sqrt(scores * topk_cls_score[k]))  # default is sqrt

        new_segments = torch.cat(new_segments)
        new_scores = torch.cat(new_scores)
        return new_segments, new_labels, new_scores


@CLASSIFIERS.register_module()
class PseudoClassifier:
    def __init__(self, pseudo_label=""):
        super().__init__()

        self.pseudo_label = pseudo_label

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)

        labels = [self.pseudo_label for _ in range(len(segments))]

        return segments, labels, scores
