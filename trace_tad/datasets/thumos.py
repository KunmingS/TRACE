import numpy as np
import math
from copy import deepcopy
from .base import SlidingWindowDataset, PaddingDataset, filter_same_annotation
from .builder import DATASETS


@DATASETS.register_module()
class ThumosSlidingDataset(SlidingWindowDataset):
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            # gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            # gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])
            gt_start = anno["frame_segment"][0] if "frame_segment" in anno else int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = anno["frame_segment"][1] if "frame_segment" in anno else int(anno["segment"][1] / video_info["duration"] * video_info["frame"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict
            # frame divided by snippet stride inside current window
            # this is only valid gt inside this window
            video_anno["gt_segments"] = video_anno["gt_segments"] - window_snippet_centers[0] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        sample = dict(
            video_name=video_name,
            data_path=self.data_path,
            window_size=self.window_size,
            # trunc window setting
            feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
            feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
            sample_stride=self.sample_stride,
            # sliding post process setting
            fps=video_info["frame"] / video_info["duration"],
            snippet_stride=self.snippet_stride,
            window_start_frame=window_snippet_centers[0],
            duration=video_info["duration"],
            offset_frames=self.offset_frames,
            # training setting
            **video_anno,
        )
        # Virtual-clip metadata (omitted for legacy physical clips). When set,
        # the pipeline reads frames from `source_video` with an offset of
        # `source_frame_offset` instead of opening data_path/<name>.mp4.
        if "source_video" in video_info:
            sample["source_video"] = video_info["source_video"]
            sample["source_frame_offset"] = video_info.get("source_frame_offset", 0)
            sample["clip_frame_count"] = video_info["frame"]
            # PTS-table reference (added in Phase 2 of the refactor in
            # docs/pts-based-frame-mapping.md). Consumed by
            # `convert_to_seconds` to map model-frame indices back to
            # clip-relative seconds without assuming CFR. Missing on
            # virtual-clip datasets prepped before the PTS upgrade — those
            # fall back to the `fps`-based path.
            if "source_pts_table" in video_info:
                sample["source_pts_table"] = video_info["source_pts_table"]
        if "cached_video" in video_info:
            sample["cached_video"] = video_info["cached_video"]
            sample["decode_frame_offset"] = 0
        results = self.pipeline(sample)
        return results


@DATASETS.register_module()
class ThumosPaddingDataset(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            
            # Check if frame indices are directly provided in the annotation
            if "frame_segment" in anno:
                # Use frame indices directly from the annotation
                gt_start = int(anno["frame_segment"][0])
                gt_end = int(anno["frame_segment"][1])
            else:
                # Convert from time to frame indices (fallback for old format)
                gt_start = math.floor(anno["segment"][0] / video_info["duration"] * video_info["frame"])
                gt_end = math.ceil(anno["segment"][1] / video_info["duration"] * video_info["frame"])
                
                # Ensure gt_end > gt_start to avoid zero duration
                if gt_end <= gt_start:
                    gt_end = gt_start + 1

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict
            video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        sample = dict(
            video_name=video_name,
            data_path=self.data_path,
            sample_stride=self.sample_stride,
            snippet_stride=self.snippet_stride,
            fps=video_info["frame"] / video_info["duration"],
            duration=video_info["duration"],
            offset_frames=self.offset_frames,
            **video_anno,
        )
        if "source_video" in video_info:
            sample["source_video"] = video_info["source_video"]
            sample["source_frame_offset"] = video_info.get("source_frame_offset", 0)
            sample["clip_frame_count"] = video_info["frame"]
            # See ThumosSlidingDataset.__getitem__ above for rationale.
            if "source_pts_table" in video_info:
                sample["source_pts_table"] = video_info["source_pts_table"]
        if "cached_video" in video_info:
            sample["cached_video"] = video_info["cached_video"]
            sample["decode_frame_offset"] = 0
        results = self.pipeline(sample)
        return results
