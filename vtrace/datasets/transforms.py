import copy
import os
import pickle
import random
import torch
import random
import pandas as pd
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


# ── merged from loading.py ──
import copy
import os
import pickle
import random
import torch
import random
import pandas as pd
import numpy as np

from .builder import PIPELINES
from torch.nn import functional as F


@PIPELINES.register_module()
class LoadFeats:
    def __init__(self, feat_format, prefix="", suffix=""):
        self.feat_format = feat_format
        self.prefix = prefix
        self.suffix = suffix
        # check feat format
        if isinstance(self.feat_format, str):
            self.check_feat_format(self.feat_format)
        elif isinstance(self.feat_format, list):
            for feat_format in self.feat_format:
                self.check_feat_format(feat_format)

    def check_feat_format(self, feat_format):
        assert feat_format in ["npy", "npz", "pt", "csv", "pkl"], print(f"not support {feat_format}")

    def read_from_tensor(self, file_path):
        feats = torch.load(file_path).float()
        return feats

    def read_from_npy(self, file_path):
        feats = np.load(file_path).astype(np.float32)
        return feats

    def read_from_npz(self, file_path):
        feats = np.load(file_path)["feats"].astype(np.float32)
        return feats

    def read_from_csv(self, file_path):
        feats = pd.read_csv(file_path, dtype="float32").to_numpy()
        feats = feats.astype(np.float32)
        return feats

    def read_from_pkl(self, file_path):
        feats = pickle.load(open(file_path, "rb"))
        feats = feats.astype(np.float32)
        return feats

    def load_single_feat(self, file_path, feat_format):
        try:
            if feat_format == "npy":
                feats = self.read_from_npy(file_path)
            elif feat_format == "npz":
                feats = self.read_from_npz(file_path)
            elif feat_format == "pt":
                feats = self.read_from_tensor(file_path)
            elif feat_format == "csv":
                feats = self.read_from_csv(file_path)
            elif feat_format == "pkl":
                feats = self.read_from_pkl(file_path)
        except:
            print("Missing data:", file_path)
            exit()
        return feats

    def __call__(self, results):
        video_name = results["video_name"]

        if isinstance(results["data_path"], str):
            file_path = os.path.join(results["data_path"], f"{self.prefix}{video_name}{self.suffix}.{self.feat_format}")
            feats = self.load_single_feat(file_path, self.feat_format)
        elif isinstance(results["data_path"], list):
            feats = []

            # check if the feat_format is a list
            if isinstance(self.feat_format, str):
                self.feat_format = [self.feat_format] * len(results["data_path"])

            for data_path, feat_format in zip(results["data_path"], self.feat_format):
                file_path = os.path.join(data_path, f"{self.prefix}{video_name}{self.suffix}.{feat_format}")
                feats.append(self.load_single_feat(file_path, feat_format))

            max_len = max([feat.shape[0] for feat in feats])
            for i in range(len(feats)):
                if feats[i].shape[0] != max_len:
                    # assume the first dimension is T
                    tmp_feat = F.interpolate(
                        torch.Tensor(feats[i]).permute(1, 0).unsqueeze(0),
                        size=max_len,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0)
                    feats[i] = tmp_feat.permute(1, 0).numpy()
            feats = np.concatenate(feats, axis=1)

        # sample the feature
        sample_stride = results.get("sample_stride", 1)
        if sample_stride > 1:
            feats = feats[::sample_stride]

        results["feats"] = feats
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"feat_format={self.feat_format}"
        return repr_str


@PIPELINES.register_module()
class SlidingWindowTrunc:
    """This is used for sliding window dataset, which will give a window start and window end in the result dict,
    and we will extract the window features, also pad to fixed length"""

    def __init__(self, with_mask=True):
        self.with_mask = with_mask

    def __call__(self, results):
        assert "window_size" in results.keys(), "should have window_size as a key"
        assert isinstance(results["feats"], torch.Tensor)
        window_size = results["window_size"]

        feats_length = results["feats"].shape[0]
        start_idx = min(results["feature_start_idx"], feats_length)
        end_idx = min(results["feature_end_idx"] + 1, feats_length)

        window_feats = results["feats"][start_idx:end_idx]
        valid_len = window_feats.shape[0]

        # if the valid window is smaller than window size, pad with -1
        if valid_len < window_size:
            pad_data = torch.zeros(window_size - valid_len, window_feats.shape[1])
            window_feats = torch.cat((window_feats, pad_data), dim=0)

        # if we need padding mask (valid is 1, pad is 0)
        if self.with_mask:
            if valid_len < window_size:
                masks = torch.cat([torch.ones(valid_len), torch.zeros(window_size - valid_len)])
            else:
                masks = torch.ones(window_size)
            results["masks"] = masks.bool()

        results["feats"] = window_feats.float()
        return results


@PIPELINES.register_module()
class RandomTrunc:
    """Crops features within a window such that they have a large overlap with ground truth segments.
    Withing the cropping ratio, the length is sampled."""

    def __init__(
        self,
        trunc_len,
        trunc_thresh,
        crop_ratio=None,
        max_num_trials=200,
        has_action=True,
        no_trunc=False,
        pad_value=0,
        channel_first=False,
    ):
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.max_num_trials = max_num_trials
        self.has_action = has_action
        self.no_trunc = no_trunc
        self.pad_value = pad_value
        self.channel_first = channel_first

    def trunc_features(self, feats, gt_segments, gt_labels, offset):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = self.trunc_len
        if feat_len <= self.trunc_len:
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
        for _ in range(self.max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = torch.as_tensor([st, ed], dtype=torch.float32)

            # compute the intersection between the sampled window and all segments
            window = window[None].repeat(num_segs, 1)
            left = torch.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = torch.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = (right - left).clamp(min=0)
            area_segs = torch.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            if self.no_trunc:
                # with at least one action and not truncating any actions
                seg_trunc_idx = (inter_ratio > 0.0) & (inter_ratio < 1.0)
                if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                    break
            elif self.has_action:
                # with at least one action
                if seg_idx.sum().item() > 0:
                    break
            else:
                # without any constraints
                break

        feats = feats[st:ed, :]  # [T,C]
        gt_segments = torch.stack((left[seg_idx], right[seg_idx]), dim=1)  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        return feats, gt_segments, gt_labels

    def pad_features(self, feats):
        feat_len = feats.shape[0]
        if feat_len < self.trunc_len:
            feats_pad = torch.ones((self.trunc_len - feat_len,) + feats.shape[1:]) * self.pad_value
            feats = torch.cat([feats, feats_pad], dim=0)
            masks = torch.cat([torch.ones(feat_len), torch.zeros(self.trunc_len - feat_len)])
            return feats, masks
        else:
            return feats, torch.ones(feat_len)

    def __call__(self, results):
        assert isinstance(results["feats"], torch.Tensor)
        offset = 0

        if self.channel_first:
            results["feats"] = results["feats"].transpose(0, 1)  # [C,T] -> [T,C]

        # truncate the features
        feats, gt_segments, gt_labels = self.trunc_features(
            results["feats"],
            results["gt_segments"],
            results["gt_labels"],
            offset,
        )

        # pad the features to the fixed length
        feats, masks = self.pad_features(feats)

        results["feats"] = feats.float()
        results["masks"] = masks.bool()
        results["gt_segments"] = gt_segments
        results["gt_labels"] = gt_labels

        if self.channel_first:
            results["feats"] = results["feats"].transpose(0, 1)  # [T,C] -> [C,T]
        return results


# ── merged from formatting.py ──
import torch
import torch.nn.functional as F
import torchvision
import scipy
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


@PIPELINES.register_module()
class ResizeFeat:
    def __init__(self, tool, channel_first=False):
        self.tool = tool
        self.channel_first = channel_first

    @torch.no_grad()
    def torchvision_align(self, feat, tscale):
        # input feat shape [C,T]
        pseudo_input = feat.unsqueeze(0).unsqueeze(3)  # [1,C,T,1]
        pseudo_bbox = torch.Tensor([[0, 0, 0, 1, feat.shape[1]]])
        # output feat shape [C,tscale]
        output = torchvision.ops.roi_align(
            pseudo_input.half().double(),
            pseudo_bbox.half().double(),
            output_size=(tscale, 1),
            aligned=True,
        ).to(pseudo_input.dtype)
        output = output.squeeze(0).squeeze(-1)
        return output

    @torch.no_grad()
    def gtad_align(self, feat):
        raise "not implement yet"

    @torch.no_grad()
    def bmn_align(self, feat, tscale, num_bin=1, num_sample_bin=3, pool_type="mean"):
        feat = feat.numpy()
        C, T = feat.shape

        # x is the temporal location corresponding to each location  in feature sequence
        x = [0.5 + ii for ii in range(T)]
        f = scipy.interpolate.interp1d(x, feat, axis=1)

        video_feature = []
        zero_sample = np.zeros(num_bin * C)
        tmp_anchor_xmin = [1.0 / tscale * i for i in range(tscale)]
        tmp_anchor_xmax = [1.0 / tscale * i for i in range(1, tscale + 1)]

        num_sample = num_bin * num_sample_bin
        for idx in range(tscale):
            xmin = max(x[0] + 0.0001, tmp_anchor_xmin[idx] * T)
            xmax = min(x[-1] - 0.0001, tmp_anchor_xmax[idx] * T)
            if xmax < x[0]:
                video_feature.append(zero_sample)
                continue
            if xmin > x[-1]:
                video_feature.append(zero_sample)
                continue

            plen = (xmax - xmin) / (num_sample - 1)
            x_new = [xmin + plen * ii for ii in range(num_sample)]
            y_new = f(x_new)
            y_new_pool = []
            for b in range(num_bin):
                tmp_y_new = y_new[:, num_sample_bin * b : num_sample_bin * (b + 1)]
                if pool_type == "mean":
                    tmp_y_new = np.mean(tmp_y_new, axis=1)
                elif pool_type == "max":
                    tmp_y_new = np.max(tmp_y_new, axis=1)
                y_new_pool.append(tmp_y_new)
            y_new_pool = np.stack(y_new_pool, axis=1).reshape(-1)
            # y_new_pool = np.reshape(y_new_pool, [-1])
            video_feature.append(y_new_pool)
        video_feature = np.stack(video_feature, axis=1)
        return torch.from_numpy(video_feature)

    @torch.no_grad()
    def torch_interpolate(self, feat, tscale):
        # input feat shape [C,T]
        feats = F.interpolate(feat.unsqueeze(0), size=tscale, mode="linear", align_corners=False).squeeze(0)
        return feats

    def __call__(self, results):
        assert "resize_length" in results.keys(), "should have resize_length as a key"
        tscale = results["resize_length"]

        if not self.channel_first:
            feats = results["feats"].permute(1, 0)  # [T,C] -> [C,T]
        else:
            feats = results["feats"]

        assert isinstance(feats, torch.Tensor)
        assert feats.ndim == 2  # [C,T]

        if self.tool == "torchvision_align":
            resized_feat = self.torchvision_align(feats, tscale)
        elif self.tool == "gtad_align":
            resized_feat = self.gtad_align(feats, tscale)
        elif self.tool == "bmn_align":
            resized_feat = self.bmn_align(feats, tscale)
        elif self.tool == "interpolate":
            resized_feat = self.torch_interpolate(feats, tscale)

        assert resized_feat.shape[0] == feats.shape[0]
        assert resized_feat.shape[1] == tscale

        if "gt_segments" in results.keys():
            # convert gt seconds to feature grid
            results["gt_segments"] = (results["gt_segments"] / results["duration"]).clamp(min=0.0, max=1.0)
            results["gt_segments"] *= tscale

        results["feats_len_ori"] = results["feats"].shape[1]  # for future usage
        if not self.channel_first:
            results["feats"] = resized_feat.permute(1, 0)  # [C,T] -> [T,C]
        else:
            results["feats"] = resized_feat
        return results


@PIPELINES.register_module()
class Padding:
    def __init__(self, length, pad_value=0, channel_first=False):
        self.length = length
        self.pad_value = pad_value
        self.channel_first = channel_first

    def __call__(self, results):
        assert "feats" in results.keys(), "should have feats as a key"
        assert results["feats"].ndim == 2, "feats should be 2 dim"

        if self.channel_first:
            feats = results["feats"].permute(1, 0)
        else:
            feats = results["feats"]

        feat_len = feats.shape[0]
        if feat_len < self.length:
            pad = torch.ones((self.length - feat_len, feats.shape[1])) * self.pad_value
            new_feats = torch.cat((feats, pad), dim=0)

            if self.channel_first:
                results["feats"] = new_feats.permute(1, 0)
            else:
                results["feats"] = new_feats

            pad_masks = torch.zeros(self.length - feat_len).bool()
            if "masks" in results.keys():
                results["masks"] = torch.cat((results["masks"], pad_masks), dim=0)
            else:
                results["masks"] = torch.cat((torch.ones(feat_len).bool(), pad_masks), dim=0)
        else:
            print(f"feature length {feat_len} is larger than padding length. Will be resized to {self.length}.")
            results["snippet_stride"] = results["snippet_stride"] * feat_len / self.length
            results["offset_frames"] = results["offset_frames"] * feat_len / self.length
            new_feats = F.interpolate(
                feats.permute(1, 0)[None],  # [b,c,t]
                size=self.length,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
            # new_feats [c,t]
            results["feats"] = new_feats if self.channel_first else new_feats.permute(1, 0)
            results["masks"] = torch.ones(self.length).bool()
        return results


@PIPELINES.register_module()
class ChannelReduction:
    """Select features along the channel dimension."""

    def __init__(self, in_channels, index):
        self.in_channels = in_channels
        self.index = index
        assert len(self.index) == 2

    def __call__(self, results):
        assert isinstance(results["feats"], torch.Tensor)
        assert results["feats"].shape[1] == self.in_channels  # [T,C]

        # select the features
        results["feats"] = results["feats"][:, self.index[0] : self.index[1]]
        return results


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
import torchvision
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
        flat_inds = np.clip(flat_inds, 0, len(vr) - 1).tolist()
        imgs = self._get_batch(vr, flat_inds, results).asnumpy()  # [N, H, W, 3]
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
class VideoColorJitter:
    """Applies consistent color jitter across all frames using numpy batch ops.
    Avoids per-frame numpy->PIL->numpy conversion overhead.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness)
        self.contrast = self._check_input(contrast)
        self.saturation = self._check_input(saturation)
        self.hue = self._check_input(hue, center=0, bound=0.5, clip_first_on_zero=False)

    @staticmethod
    def _check_input(value, center=1, bound=float("inf"), clip_first_on_zero=True):
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"Value {value} must be non-negative.")
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        return value

    def __call__(self, results):
        imgs = results["imgs"]
        if not imgs:
            return results

        # sample parameters once for all frames
        brightness_factor = random.uniform(*self.brightness) if self.brightness else None
        contrast_factor = random.uniform(*self.contrast) if self.contrast else None
        saturation_factor = random.uniform(*self.saturation) if self.saturation else None
        hue_factor = random.uniform(*self.hue) if self.hue else None

        # random order
        fn_idx = list(range(4))
        random.shuffle(fn_idx)

        new_imgs = []
        for img in imgs:
            is_numpy = isinstance(img, np.ndarray)
            if not is_numpy:
                img = np.array(img)
            img = img.astype(np.float32)

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = img * brightness_factor
                elif fn_id == 1 and contrast_factor is not None:
                    mean = img.mean(axis=(0, 1), keepdims=True)
                    img = (img - mean) * contrast_factor + mean
                elif fn_id == 2 and saturation_factor is not None:
                    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
                    gray = gray[..., np.newaxis]
                    img = (img - gray) * saturation_factor + gray
                elif fn_id == 3 and hue_factor is not None and hue_factor != 0:
                    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
                    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[..., 0] = (hsv[..., 0] + hue_factor * 180) % 180
                    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

            new_imgs.append(np.clip(img, 0, 255).astype(np.uint8))

        results["imgs"] = new_imgs
        return results


@PIPELINES.register_module()
class VideoImgAug:
    """Applies random Gaussian blur to a clip with 50% probability.

    Uses torchvision instead of the deprecated imgaug library.
    """

    def __init__(self, transforms=None, p=0.5, sigma=(0.1, 3.0)):
        self.p = p
        self.sigma = sigma

    def __call__(self, results):
        if random.random() > self.p:
            return results
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        # kernel size must be odd and large enough for the sigma
        kernel_size = int(2 * round(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(kernel_size, 3)
        blur = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        results["imgs"] = [
            np.array(blur(Image.fromarray(img if isinstance(img, np.ndarray) else np.array(img))))
            for img in results["imgs"]
        ]
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


@PIPELINES.register_module()
class VideoNormalize:
    """Pixel normalisation transform (replaces ActionDataPreprocessor).
    Applied BEFORE or as part of the pipeline if not done in BackboneWrapper.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, results):
        imgs = results["imgs"]
        if isinstance(imgs, np.ndarray):
            results["imgs"] = (imgs.astype(np.float32) - self.mean) / self.std
        elif isinstance(imgs, torch.Tensor):
            mean = torch.tensor(self.mean, device=imgs.device).reshape(1, 3, 1, 1, 1)
            std = torch.tensor(self.std, device=imgs.device).reshape(1, 3, 1, 1, 1)
            results["imgs"] = (imgs.float() - mean) / std
        return results
