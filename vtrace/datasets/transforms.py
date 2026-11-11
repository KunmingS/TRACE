import os
import random
import torch
import random
import numpy as np

from .builder import PIPELINES
from torch.nn import functional as F


@PIPELINES.register_module()
class PrepareVideoInfo:
    def __init__(self, format="mp4", modality="RGB", prefix=""):
        self.format = format
        self.modality = modality
        self.prefix = prefix

    def __call__(self, results):
        results["modality"] = self.modality
        # Decode proxy: a downscaled, frame-aligned copy of the whole source
        # video. Frame i of the proxy is frame i of the source, so
        # `source_frame_offset` keeps applying unchanged — deliberately no
        # `decode_frame_offset` override here.
        if results.get("proxy_video"):
            results["filename"] = results["proxy_video"]
        # Legacy per-window clip cache: no longer written, still honored so
        # datasets prepared before the proxy refactor keep loading. Those clips
        # start at frame 0, hence the offset reset.
        elif results.get("cached_video"):
            results["filename"] = results["cached_video"]
            results["decode_frame_offset"] = 0
        # Virtual clips: read frames directly from the source video, not from a
        # pre-extracted clip file. `source_video` is an absolute path written by
        # data_prep.py.
        elif results.get("source_video"):
            results["filename"] = results["source_video"]
        else:
            results["filename"] = os.path.join(
                results["data_path"],
                self.prefix + results["video_name"] + "." + self.format,
            )
        return results


@PIPELINES.register_module()
class LoadSnippetFrames:
    """Load the snippet frame, the output should follows the format:
    snippet_num x channel x clip_len x height x width
    """

    def __init__(
        self,
        clip_len,
        frame_interval=1,
        method="resize",
        trunc_len=None,
        trunc_thresh=None,
        crop_ratio=None,
    ):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.method = method  # resize or padding or sliding window
        # random_trunc settings
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio

    def random_trunc(self, feats, trunc_len, gt_segments, gt_labels, offset=0, max_num_trials=200):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = trunc_len
        if feat_len <= trunc_len:
            if self.crop_ratio == None:  # do nothing
                return feats, gt_segments, gt_labels
            else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                # corner case
                if feat_len == trunc_len:
                    return feats, gt_segments, gt_labels

        # try a few times till a valid truncation with at least one action
        for _ in range(max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = np.array([st, ed], dtype=np.float32)

            # compute the intersection between the sampled window and all segments
            window = np.repeat(window[None, :], num_segs, axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = np.clip(right - left, a_min=0, a_max=None)
            area_segs = np.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            # with at least one action
            if seg_idx.sum().item() > 0:
                break

        feats = feats[st:ed]
        gt_segments = np.stack((left[seg_idx], right[seg_idx]), axis=1)  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        return feats, gt_segments, gt_labels

    def __call__(self, results):
        assert "total_frames" in results.keys(), "should have total_frames as a key"
        total_frames = results["total_frames"]
        fps = results["avg_fps"]

        if self.method == "resize":
            assert "resize_length" in results.keys(), "should have resize_length as a key"
            snippet_num = results["resize_length"]
            snippet_stride = total_frames / snippet_num
            snippet_center = np.arange(
                snippet_stride / 2 - 0.5,
                total_frames + snippet_stride / 2 - 0.5,
                snippet_stride,
            )
            masks = torch.ones(results["resize_length"]).bool()

            # don't forget to resize the ground truth segments
            if "gt_segments" in results.keys():
                # convert gt seconds to feature grid
                results["gt_segments"] = np.clip(results["gt_segments"] / results["duration"], 0.0, 1.0)
                results["gt_segments"] *= results["resize_length"]

        elif self.method == "random_trunc":
            snippet_num = self.trunc_len
            snippet_center = np.arange(0, total_frames, results["snippet_stride"])

            # trunc the snippet_center
            snippet_center, gt_segments, gt_labels = self.random_trunc(
                snippet_center,
                trunc_len=snippet_num,
                gt_segments=results["gt_segments"],
                gt_labels=results["gt_labels"],
            )

            # update the gt_segments
            results["gt_segments"] = gt_segments
            results["gt_labels"] = gt_labels

            # pad the snippet_center
            if len(snippet_center) < snippet_num:
                valid_len = len(snippet_center)
                snippet_center = np.pad(snippet_center, (0, snippet_num - valid_len), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(snippet_num - valid_len)]).bool()
            else:
                masks = torch.ones(snippet_num).bool()

        elif self.method == "sliding_window":
            snippet_num = results["window_size"]
            snippet_center = np.arange(0, total_frames, results["snippet_stride"])

            start_idx = min(results["feature_start_idx"], len(snippet_center))
            end_idx = min((results["feature_end_idx"] + 1), len(snippet_center))

            snippet_center = snippet_center[start_idx:end_idx]

            if len(snippet_center) < snippet_num:
                valid_len = len(snippet_center)
                snippet_center = np.pad(snippet_center, (0, snippet_num - valid_len), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(snippet_num - valid_len)]).bool()
            else:
                masks = torch.ones(snippet_num).bool()
        elif self.method == "padding":
            raise NotImplementedError

        # extend snippet center to a clip
        clip_idxs = np.arange(-(self.clip_len // 2), self.clip_len // 2)
        frame_idxs = snippet_center[:, None] + self.frame_interval * clip_idxs[None, :]  # [snippet_num, clip_len]

        # truncate to [0, total_frames-1], and round to int
        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round()

        assert frame_idxs.shape[0] == snippet_num, "snippet center number should be equal to snippet number"
        assert frame_idxs.shape[1] == self.clip_len, "snippet length should be equal to clip length"

        results["frame_inds"] = frame_idxs.astype(int)
        results["num_clips"] = snippet_num
        results["clip_len"] = self.clip_len
        results["masks"] = masks
        return results


@PIPELINES.register_module()
class LoadFrames:
    def __init__(
        self,
        num_clips=1,
        scale_factor=1,
        method="resize",
        trunc_len=None,
        trunc_thresh=None,
        crop_ratio=None,
    ):
        self.num_clips = num_clips
        self.scale_factor = scale_factor  # multiply by the frame number, if backbone has downsampling
        self.method = method  # resize or padding or random_trunc or sliding_window
        # random_trunc settings
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio

    def random_trunc(self, feats, trunc_len, gt_segments, gt_labels, offset=0, max_num_trials=200):
        # return feats, gt_segments, gt_labels
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = trunc_len
        if feat_len <= trunc_len:
            if self.crop_ratio == None:  # do nothing
                return feats, gt_segments, gt_labels
            else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                # corner case
                if feat_len == trunc_len:
                    return feats, gt_segments, gt_labels

        # try a few times till a valid truncation with at least one action
        for _ in range(max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = np.array([st, ed], dtype=np.float32)

            # compute the intersection between the sampled window and all segments
            window = np.repeat(window[None, :], num_segs, axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = np.clip(right - left, a_min=0, a_max=None)
            area_segs = np.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            # with at least one action
            if seg_idx.sum().item() > 0:
                break

        feats = feats[st:ed]
        gt_segments = np.stack((left[seg_idx], right[seg_idx]), axis=1)  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        return feats, gt_segments, gt_labels

    def __call__(self, results):
        assert "total_frames" in results.keys(), "should have total_frames as a key"
        total_frames = results["total_frames"]
        fps = results["avg_fps"]

        if self.method == "resize":
            assert "resize_length" in results.keys(), "should have resize_length as a key"
            frame_num = results["resize_length"] * self.scale_factor
            frame_stride = total_frames / frame_num
            frame_idxs = np.arange(
                frame_stride / 2 - 0.5,
                total_frames + frame_stride / 2 - 0.5,
                frame_stride,
            )
            masks = torch.ones(results["resize_length"]).bool()  # should not multiply by scale_factor

            # don't forget to resize the ground truth segments
            if "gt_segments" in results.keys():
                # convert gt seconds to feature grid
                results["gt_segments"] = np.clip(results["gt_segments"] / results["duration"], 0.0, 1.0)
                results["gt_segments"] *= results["resize_length"]

        elif self.method == "random_trunc":
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            frame_num = self.trunc_len * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)

            # trunc the frame_idxs
            frame_idxs, gt_segments, gt_labels = self.random_trunc(
                frame_idxs,
                trunc_len=frame_num,
                gt_segments=results["gt_segments"] * self.scale_factor,  # gt segment should be mapped to frame level
                gt_labels=results["gt_labels"],
            )
            results["gt_segments"] = gt_segments / self.scale_factor  # convert back to original scale
            results["gt_labels"] = gt_labels

            # pad the frame_idxs
            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(self.trunc_len - valid_len)]).bool()
            else:
                masks = torch.ones(self.trunc_len).bool()

        elif self.method == "sliding_window":
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            window_size = results["window_size"]
            frame_num = window_size * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)

            start_idx = min(results["feature_start_idx"] * self.scale_factor, len(frame_idxs))
            end_idx = min((results["feature_end_idx"] + 1) * self.scale_factor, len(frame_idxs))

            frame_idxs = frame_idxs[start_idx:end_idx]

            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(window_size - valid_len)]).bool()
            else:
                masks = torch.ones(window_size).bool()

        elif self.method == "padding":
            raise NotImplementedError

        # truncate to [0, total_frames-1], and round to int
        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round()

        assert frame_idxs.shape[0] == frame_num, "snippet center number should be equal to snippet number"

        results["frame_inds"] = frame_idxs.astype(int)
        results["num_clips"] = self.num_clips
        results["clip_len"] = frame_num // self.num_clips
        results["masks"] = masks
        return results


@PIPELINES.register_module()
class Interpolate:
    def __init__(self, keys, size=128, mode="linear"):
        self.keys = keys
        self.size = size
        self.mode = mode

    def __call__(self, results):
        for key in self.keys:
            if results[key].shape[2:] != self.size:
                results[key] = F.interpolate(
                    results[key],
                    size=self.size,
                    mode=self.mode,
                    align_corners=False,
                )
        return results


# ── merged from formatting.py ──
import torch
import torch.nn.functional as F
import numpy as np
from collections.abc import Sequence
from einops import rearrange, reduce

from .builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, Sequence):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@PIPELINES.register_module()
class Collect:
    def __init__(
        self,
        inputs,
        keys=[],
        meta_keys=[
            "video_name",
            "data_path",
            "fps",
            "duration",
            "snippet_stride",
            "window_start_frame",
            "resize_length",
            "window_size",
            "offset_frames",
            # PTS-based timestamp ↔ frame mapping (Phase 2 of the refactor in
            # pts-based-frame-mapping.md (archived)). Present whenever the dataset
            # carries `source_pts_table`; consumed by `convert_to_seconds`
            # in the post-processing path. Missing on legacy CFR-only
            # datasets, which fall back to `fps`.
            "source_pts_table",
            "source_frame_offset",
            "clip_frame_count",
            "proxy_video",
            "cached_video",
        ],
    ):
        self.inputs = inputs
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}

        # input key
        data["inputs"] = results[self.inputs]  # [C,T]

        # AutoAugment key: gt_segments, gt_labels, masks
        for key in self.keys:
            if key == "masks" and key not in results.keys():
                results["masks"] = torch.ones(data["inputs"].shape[-1]).bool()
            data[key] = results[key]

        # meta keys
        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                if key in results.keys():
                    meta[key] = results[key]
            data["metas"] = meta

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"keys={self.keys}, meta_keys={self.meta_keys}, "


@PIPELINES.register_module()
class ConvertToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys})"


@PIPELINES.register_module()
class Rearrange:
    def __init__(self, keys, ops, **kwargs):
        self.keys = keys
        self.ops = ops
        self.kwargs = kwargs

    def __call__(self, results):
        for key in self.keys:
            results[key] = rearrange(results[key], self.ops, **self.kwargs)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys}ops={self.ops})"


@PIPELINES.register_module()
class Reduce:
    def __init__(self, keys, ops, reduction):
        self.keys = keys
        self.ops = ops
        self.reduction = reduction

    def __call__(self, results):
        for key in self.keys:
            results[key] = reduce(results[key], self.ops, reduction=self.reduction)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys}ops={self.ops})reduction={self.reduction}"


# ── merged from video_transforms.py ──
"""
Pure decord + torchvision replacements for mmaction video pipeline steps.
Replaces: mmaction.DecordInit, DecordDecode, Resize, RandomResizedCrop,
          CenterCrop, Flip, ImgAug, ColorJitter, FormatShape.
"""
import os
import random
import warnings
from collections import OrderedDict

import numpy as np
import torch
import cv2
import decord
import torchvision.transforms.functional as TF
from PIL import Image

from .builder import PIPELINES


class _VideoReaderCache:
    """Per-process LRU cache of decord.VideoReader instances.

    Virtual clips read frame ranges directly from the original source video
    rather than from pre-extracted clip files. Without caching, every
    __getitem__ call would reopen the source (parsing moov atom etc.). With
    caching + DataLoader's persistent_workers=True, hot source videos stay
    open across samples and across epochs.

    Each PyTorch worker is a separate process, so each gets its own cache.
    """

    def __init__(self, maxsize: int = None):
        # Each cached reader holds decoder state for a whole source video; on
        # long videos that is several GB per entry, so a worker with the default
        # 8 entries can reach ~50 GB RSS. On hosts with less RAM (or when you
        # want more workers instead of a bigger per-worker cache), shrink it via
        # TRACE_VR_CACHE. Default is unchanged.
        if maxsize is None:
            maxsize = int(os.environ.get("TRACE_VR_CACHE", "8"))
        self._cache: "OrderedDict[tuple[str, int, int, int], decord.VideoReader]" = OrderedDict()
        self._max = max(1, maxsize)

    @staticmethod
    def _key(path: str, num_threads: int, width: int, height: int):
        return (os.path.abspath(path), int(width), int(height), int(num_threads))

    def evict(self, path: str, num_threads: int, width: int = -1, height: int = -1) -> None:
        """Drop a reader whose decoder state went bad (e.g. decord EOF-retry
        error) so the next get() reopens the file from scratch."""
        old = self._cache.pop(self._key(path, num_threads, width, height), None)
        del old

    def get(self, path: str, num_threads: int, width: int = -1, height: int = -1) -> decord.VideoReader:
        key = self._key(path, num_threads, width, height)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if len(self._cache) >= self._max:
            _, old = self._cache.popitem(last=False)
            del old
        vr = decord.VideoReader(path, width=width, height=height, num_threads=num_threads)
        self._cache[key] = vr
        return vr


_video_reader_cache = _VideoReaderCache()


@PIPELINES.register_module()
class VideoInit:
    """Open the video for this sample.

    For virtual clips (sample dict contains `clip_frame_count`), `total_frames`
    is the clip's logical length, not the source video's full length, so that
    downstream LoadFrames/LoadSnippetFrames sample within the clip's span. The
    actual seek to source-frame coordinates happens in VideoDecode by adding
    `source_frame_offset` to the clip-local indices.
    """

    def __init__(self, num_threads: int = 4, resize=None, width: int = -1, height: int = -1):
        self.num_threads = num_threads
        if resize is not None:
            if isinstance(resize, int):
                height = width = resize
            else:
                height, width = resize
        self.width = int(width)
        self.height = int(height)

    def __call__(self, results):
        filename = results["filename"]
        vr = _video_reader_cache.get(filename, self.num_threads, width=self.width, height=self.height)
        results["video_reader"] = vr
        results["video_reader_spec"] = (filename, self.num_threads, self.width, self.height)
        if "clip_frame_count" in results:
            results["total_frames"] = int(results["clip_frame_count"])
        else:
            results["total_frames"] = len(vr)
        results["avg_fps"] = vr.get_avg_fps()
        return results


@PIPELINES.register_module()
class VideoTemporalAugment:
    """Random temporal speed augmentation applied after LoadFrames, before VideoDecode.

    Resamples frame_inds to simulate playback speed changes, and scales
    gt_segments proportionally.

    Args:
        speed_range: [min_speed, max_speed], e.g. [0.8, 1.2].
        p: Probability of applying the augmentation.
    """

    def __init__(self, speed_range=(0.8, 1.2), p=0.5):
        self.speed_range = speed_range
        self.p = p

    def __call__(self, results):
        if random.random() > self.p:
            return results

        frame_inds = results.get("frame_inds")
        if frame_inds is None or len(frame_inds) == 0:
            return results

        speed = random.uniform(*self.speed_range)
        if abs(speed - 1.0) < 1e-6:
            return results

        total_frames = results.get("total_frames", None)
        orig_inds = frame_inds.flatten().astype(np.float64)
        num_frames = len(orig_inds)

        # Resample frame indices around the center at the new speed.
        # speed > 1 → cover a wider source range (faster playback)
        # speed < 1 → cover a narrower source range (slower playback)
        center = (orig_inds[0] + orig_inds[-1]) / 2.0
        half_span = (orig_inds[-1] - orig_inds[0]) / 2.0
        new_half = half_span * speed
        new_start = center - new_half
        new_end = center + new_half

        new_inds = np.linspace(new_start, new_end, num_frames)

        # Clamp to valid frame range
        max_idx = (total_frames - 1) if total_frames is not None else orig_inds.max()
        new_inds = np.clip(new_inds, 0, max_idx)
        new_inds = np.round(new_inds).astype(frame_inds.dtype)

        results["frame_inds"] = new_inds.reshape(frame_inds.shape)

        # Scale gt_segments by inverse speed factor (1/speed)
        if "gt_segments" in results and results["gt_segments"] is not None:
            gt_segments = results["gt_segments"].astype(np.float64)
            # The temporal origin shifts and scales. Transform segment
            # boundaries relative to the original frame range.
            orig_start = orig_inds[0]
            orig_span = orig_inds[-1] - orig_inds[0]
            if orig_span > 0:
                # Normalize segments to [0,1] in the original span, then
                # map to the new span.  Since the *content* that was at
                # position t now appears at position t/speed, we scale
                # by 1/speed relative to the temporal center.
                seg_center = (gt_segments[:, 0] + gt_segments[:, 1]) / 2.0
                seg_half = (gt_segments[:, 1] - gt_segments[:, 0]) / 2.0

                # Scale segment duration and position around the window center
                window_center = orig_span / 2.0
                new_seg_center = window_center + (seg_center - window_center) / speed
                new_seg_half = seg_half / speed

                gt_segments[:, 0] = new_seg_center - new_seg_half
                gt_segments[:, 1] = new_seg_center + new_seg_half

                # Clamp to the output temporal range
                output_len = num_frames  # after resampling, temporal length is preserved
                gt_segments = np.clip(gt_segments, 0, output_len)

            results["gt_segments"] = gt_segments.astype(np.float32)

        return results


# Frames held back from the end of a file by VideoDecode; see the comment in
# VideoDecode.__call__ for why this exists rather than a try/except alone.
EOF_GUARD = 3
EOF_GUARD_RETRY = 30


@PIPELINES.register_module()
class VideoDecode:
    """Decode the chosen frames into image arrays.

    For virtual clips, `frame_inds` are clip-local; `source_frame_offset`
    translates them into source-video coordinates before `get_batch`. The
    VideoReader is removed from `results` (workers don't need to ship it
    downstream) but kept alive in the per-process cache.
    """

    def __call__(self, results):
        frame_inds = results["frame_inds"]
        vr = results["video_reader"]
        flat_inds = frame_inds.flatten()
        offset = int(results.get("decode_frame_offset", results.get("source_frame_offset", 0)))
        if offset:
            flat_inds = flat_inds + offset
        # clamp to valid source-frame range
        # Keep clear of the last few frames. decord raises
        #   DECORDError: Unable to handle EOF because it takes too long to
        #   retrieve last few frames
        # from get_batch when a requested index sits at the very end of a file,
        # but only under concurrent load -- the same read succeeds in isolation at
        # any num_threads, on a proxy that decodes end to end and reports exactly
        # the source frame count. It is not reproducible on demand and it is not
        # bad data, so it cannot be fixed by validating the cache.
        #
        # It has to be avoided rather than caught, because the sampler guarantees
        # it will be hit: the final sliding window starts at
        # `max(0, snippet_num - window_size)` and therefore always includes the
        # last frame, so every video reaches this every epoch. When it fires it
        # takes down the whole run, and on a corpus whose first evaluation pass
        # comes after epoch 0 it lands before any checkpoint exists -- which makes
        # restart-on-crash loop forever instead of recovering.
        #
        # EOF_GUARD frames of a 30 fps video is 0.1 s at the tail of a window that
        # is 25.6 s long, duplicated from the last frame kept.
        last = len(vr) - 1
        flat_inds = np.clip(flat_inds, 0, max(0, last - EOF_GUARD))
        try:
            imgs = self._get_batch(vr, flat_inds.tolist(), results).asnumpy()  # [N, H, W, 3]
        except Exception:
            # Belt and braces: if it still fires, back off further once rather
            # than losing the run. A window decoded 1 s short of where it asked
            # is worth far more than a dead epoch.
            flat_inds = np.clip(flat_inds, 0, max(0, last - EOF_GUARD_RETRY))
            imgs = self._get_batch(vr, flat_inds.tolist(), results).asnumpy()
        imgs = imgs.reshape(*frame_inds.shape, *imgs.shape[1:])  # [..., H, W, 3]
        results["imgs"] = list(imgs) if imgs.ndim > 3 else [imgs]
        del results["video_reader"]
        results.pop("video_reader_spec", None)
        return results

    # decord 0.6 intermittently raises "Unable to handle EOF because it takes
    # too long to retrieve last few frames" from a long-lived, multi-threaded
    # reader after many seeks (dmlc/decord#150/#283). The file is fine; the
    # reader's decoder state is not. Reopen the file and retry, falling back
    # to single-threaded decode on the last attempt.
    RETRIES = 2

    def _get_batch(self, vr, flat_inds, results):
        spec = results.get("video_reader_spec")
        last_err = None
        for attempt in range(self.RETRIES + 1):
            try:
                return vr.get_batch(flat_inds)
            except decord.DECORDError as e:
                last_err = e
                if spec is None or attempt == self.RETRIES:
                    break
                filename, num_threads, width, height = spec
                _video_reader_cache.evict(filename, num_threads, width=width, height=height)
                if attempt == self.RETRIES - 1:
                    num_threads = 1  # last try: single-threaded decoder
                vr = _video_reader_cache.get(filename, num_threads, width=width, height=height)
                warnings.warn(
                    f"decord get_batch failed on {filename} (attempt {attempt + 1}/{self.RETRIES + 1}, "
                    f"reopening with num_threads={num_threads}): {str(e).splitlines()[-1][:160]}"
                )
        raise last_err


def _resize_img(img, scale):
    """Resize image. scale=(-1, 256) means shorter side to 256."""
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
    else:
        w, h = img.size

    if isinstance(scale, int):
        short, long = min(h, w), max(h, w)
        new_short = scale
        new_long = int(long * new_short / short)
        new_h, new_w = (new_short, new_long) if h <= w else (new_long, new_short)
    elif scale[0] == -1:
        short, long = min(h, w), max(h, w)
        new_short = scale[1]
        new_long = int(long * new_short / short)
        new_h, new_w = (new_short, new_long) if h <= w else (new_long, new_short)
    elif scale[1] == -1:
        short, long = min(h, w), max(h, w)
        new_short = scale[0]
        new_long = int(long * new_short / short)
        new_h, new_w = (new_short, new_long) if h <= w else (new_long, new_short)
    else:
        new_h, new_w = scale[0], scale[1]

    if isinstance(img, np.ndarray):
        pil = Image.fromarray(img)
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        return np.array(pil)
    else:
        return TF.resize(img, (new_h, new_w))


@PIPELINES.register_module()
class VideoResize:
    """Replaces mmaction.Resize. scale=(-1, 256) resizes shorter side to 256."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, results):
        imgs = results["imgs"]
        results["imgs"] = [_resize_img(img, self.scale) for img in imgs]
        return results


@PIPELINES.register_module()
class VideoBatchResize:
    """Batched resize using cv2 — directly resizes all frames to (H, W)
    without preserving aspect ratio. Much faster than per-frame PIL conversion.

    Args:
        scale: Target (height, width) tuple, e.g. (224, 224).
        interpolation: cv2 interpolation flag. Default INTER_LINEAR.
    """

    def __init__(self, scale, interpolation=cv2.INTER_LINEAR):
        if isinstance(scale, int):
            self.scale = (scale, scale)
        else:
            self.scale = tuple(scale)
        self.interpolation = interpolation

    def __call__(self, results):
        imgs = results["imgs"]
        th, tw = self.scale

        # fast path: if every frame is already the right size, skip
        if (
            isinstance(imgs[0], np.ndarray)
            and all(img.shape[0] == th and img.shape[1] == tw for img in imgs)
        ):
            return results

        results["imgs"] = [
            cv2.resize(img, (tw, th), interpolation=self.interpolation)
            for img in imgs
        ]
        return results


@PIPELINES.register_module()
class VideoRandomResizedCrop:
    """Replaces mmaction.RandomResizedCrop.
    Applies the same random crop to all frames in the clip.
    """

    def __init__(self, area_range=(0.08, 1.0), aspect_ratio_range=(3 / 4, 4 / 3)):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range

    def __call__(self, results):
        imgs = results["imgs"]
        if not imgs:
            return results

        if isinstance(imgs[0], np.ndarray):
            h, w = imgs[0].shape[:2]
        else:
            w, h = imgs[0].size

        area = h * w
        for _ in range(10):
            target_area = random.uniform(*self.area_range) * area
            ar = random.uniform(*self.aspect_ratio_range)
            new_w = int(round((target_area * ar) ** 0.5))
            new_h = int(round((target_area / ar) ** 0.5))
            if new_w <= w and new_h <= h:
                x = random.randint(0, w - new_w)
                y = random.randint(0, h - new_h)
                results["imgs"] = [
                    (img[y:y+new_h, x:x+new_w] if isinstance(img, np.ndarray)
                     else TF.crop(img, y, x, new_h, new_w))
                    for img in imgs
                ]
                return results

        # Fallback: center crop
        x = (w - min(w, h)) // 2
        y = (h - min(w, h)) // 2
        s = min(w, h)
        results["imgs"] = [
            (img[y:y+s, x:x+s] if isinstance(img, np.ndarray)
             else TF.center_crop(img, s))
            for img in imgs
        ]
        return results


@PIPELINES.register_module()
class VideoCenterCrop:
    """Replaces mmaction.CenterCrop."""

    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

    def __call__(self, results):
        imgs = results["imgs"]
        ch, cw = self.crop_size

        def _crop(img):
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                y = (h - ch) // 2
                x = (w - cw) // 2
                return img[y:y+ch, x:x+cw]
            else:
                return TF.center_crop(img, self.crop_size)

        results["imgs"] = [_crop(img) for img in imgs]
        return results


@PIPELINES.register_module()
class VideoFlip:
    """Replaces mmaction.Flip. Applies uniform random flip to all frames."""

    def __init__(self, flip_ratio: float = 0.5, direction: str = "horizontal"):
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results):
        if random.random() < self.flip_ratio:
            imgs = results["imgs"]
            if self.direction == "horizontal":
                results["imgs"] = [
                    (img[:, ::-1].copy() if isinstance(img, np.ndarray)
                     else TF.hflip(img))
                    for img in imgs
                ]
        return results


@PIPELINES.register_module()
class VideoRotate:
    """Random in-plane rotation applied UNIFORMLY across all frames of a clip
    (one angle per clip → temporally consistent).

    Label-preserving for top-down arena recordings (e.g. CalMS21): the behaviour
    class (attack/investigation/mount) does not depend on arena orientation, so an
    arbitrary rotation is a free source of viewpoint diversity. Purely spatial — it
    leaves frame_inds / gt_segments untouched. Apply AFTER VideoBatchResize so frames
    are already square; exposed corners are filled by reflection (no black bands).

    Args:
        max_angle: angle sampled uniformly from [-max_angle, max_angle] degrees.
        p: probability of applying.
    """

    def __init__(self, max_angle: float = 180.0, p: float = 0.5):
        self.max_angle = float(max_angle)
        self.p = float(p)

    def __call__(self, results):
        if random.random() > self.p:
            return results
        imgs = results.get("imgs")
        if not imgs:
            return results
        angle = random.uniform(-self.max_angle, self.max_angle)
        h, w = imgs[0].shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        results["imgs"] = [
            cv2.warpAffine(np.ascontiguousarray(im), M, (w, h),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            for im in imgs
        ]
        return results


@PIPELINES.register_module()
class VideoTrivialAugment:
    """TrivialAugmentWide applied ONCE per clip — one sampled op+magnitude across ALL
    frames (temporal consistency). FERAL's small-data regularizer; subsumes color
    jitter. Operates on the list of [H,W,3] 0-255 frames by stacking to a uint8
    [T,3,H,W] batch so torchvision v2 samples a single op for the whole clip.
    """

    def __init__(self, num_magnitude_bins=31):
        from torchvision.transforms import v2

        self._aug = v2.TrivialAugmentWide(num_magnitude_bins=num_magnitude_bins)

    def __call__(self, results):
        imgs = results["imgs"]
        if not imgs:
            return results
        arr = np.stack([np.ascontiguousarray(im) for im in imgs])  # [T,H,W,3]
        orig_dtype = arr.dtype
        t = torch.from_numpy(arr.astype(np.uint8)).permute(0, 3, 1, 2).contiguous()  # [T,3,H,W] uint8
        t = self._aug(t)
        out = t.permute(0, 2, 3, 1).contiguous().numpy().astype(orig_dtype)
        results["imgs"] = [out[i] for i in range(out.shape[0])]
        return results


@PIPELINES.register_module()
class VideoFormatShape:
    """Replaces mmaction.FormatShape.
    Converts list of [H,W,3] numpy arrays -> [N, 3, T, H, W] tensor (NCTHW).
    """

    def __init__(self, input_format: str = "NCTHW"):
        self.input_format = input_format

    def __call__(self, results):
        imgs = results["imgs"]
        if isinstance(imgs[0], np.ndarray):
            imgs = np.stack(imgs, axis=0)  # [T, H, W, 3]
            imgs = imgs.transpose(3, 0, 1, 2)  # [3, T, H, W]
            imgs = imgs[np.newaxis]  # [1, 3, T, H, W]
        elif isinstance(imgs[0], torch.Tensor):
            imgs = torch.stack(imgs, dim=0)  # [T, C, H, W]
            imgs = imgs.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
        results["imgs"] = imgs
        return results

