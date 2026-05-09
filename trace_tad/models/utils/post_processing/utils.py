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
    cached as ``<video>.pts.npy`` by ``data_prep._load_or_build_pts``.
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
       ``docs/pts-based-frame-mapping.md``.
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
