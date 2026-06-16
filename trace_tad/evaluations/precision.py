import json
import os
import numpy as np
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from .builder import EVALUATORS, remove_duplicate_annotations


def _load_clip_pts(video_info):
    """Build the clip-relative PTS array (seconds) for one dataset entry.

    Returns ``None`` if the entry has no ``source_pts_table`` (legacy CFR
    dataset prepped before the PTS upgrade — those keep using the
    ``eval_fps`` fallback).
    """
    pts_path = video_info.get("source_pts_table")
    if not pts_path or not os.path.isfile(pts_path):
        return None
    try:
        full = np.load(pts_path).astype(np.float64, copy=False)
    except Exception:
        return None
    offset = int(video_info.get("source_frame_offset", 0))
    n = int(video_info.get("frame", len(full) - offset))
    end = min(offset + n, len(full))
    if end <= offset:
        return None
    sliced = full[offset:end]
    return sliced - sliced[0]

@EVALUATORS.register_module()
class Precision:
    def __init__(
        self,
        ground_truth_filename,
        prediction_filename,
        subset,
        tiou_thresholds,
        top_k=None,
        blocked_videos=None,
        ignore_labels=None,
        thread=16,
        gt_fps=30.0,
        eval_fps=30.0,
        score_threshold=None,
    ):
        super().__init__()

        if not ground_truth_filename:
            raise IOError("Please input a valid ground truth file.")
        if not prediction_filename:
            raise IOError("Please input a valid prediction file.")

        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.top_k = top_k
        self.gt_fields = ["database"]
        self.pred_fields = ["results"]
        self.thread = thread
        self.gt_fps = float(gt_fps)
        self.eval_fps = float(eval_fps)
        # Score threshold for the reported precision/recall/F1. When None, the
        # F1-optimal global threshold found by the sweep is used (and logged),
        # instead of the historical ~0 cutoff that silently inflated recall.
        self.score_threshold = None if score_threshold is None else float(score_threshold)
        # Cache for the aligned all-frame score/GT matrices (built lazily).
        self._frame_matrices = None

        if blocked_videos is None:
            self.blocked_videos = list()
        else:
            with open(blocked_videos) as json_file:
                self.blocked_videos = json.load(json_file)

        self.ground_truth_filename = ground_truth_filename

        self._import_ground_truth(ground_truth_filename)

        # Create label encoding for faster comparisons
        self._create_label_encoding()

        self.prediction = self._import_prediction(prediction_filename)

    def _import_ground_truth(self, ground_truth_filename):
        with open(ground_truth_filename, "r") as fobj:
            gt_data = json.load(fobj)["database"]

        self.gt_anno = {}
        self.gt_segments = {}
        self.video_frames = {}
        # PTS-aware mapping: per-clip clip-relative PTS array (or None for
        # legacy CFR datasets). Built once at GT load so per-prediction
        # workers can read it without holding any locks.
        self.clip_pts = {}

        for clip_name in gt_data.keys():
            if gt_data[clip_name]["subset"] != self.subset:
                continue

            # Use per-video frame count from annotation file (after fps conversion).
            # Fallback: estimate from duration * eval_fps.
            video_info = gt_data[clip_name]
            if "frame" in video_info and video_info["frame"] is not None:
                num_frames = int(video_info["frame"])
            else:
                num_frames = int(round(float(video_info["duration"]) * self.eval_fps))

            self.video_frames[clip_name] = num_frames
            self.clip_pts[clip_name] = _load_clip_pts(video_info)
            # Multi-label: each frame stores a SET of labels (empty set = background)
            clip_behavior_list = [set() for _ in range(num_frames)]
            segments = []

            for anno in gt_data[clip_name]["annotations"]:
                label = anno["label"]

                # Prefer frame_segment if present (already aligned to eval_fps).
                if "frame_segment" in anno and anno["frame_segment"] is not None:
                    start, end = int(anno["frame_segment"][0]), int(anno["frame_segment"][1])
                else:
                    # Fallback: convert seconds segment into eval_fps frames.
                    t0, t1 = float(anno["segment"][0]), float(anno["segment"][1])
                    if t1 < t0:
                        t0, t1 = t1, t0
                    start = int(math.floor(t0 * self.eval_fps))
                    end = int(math.ceil(t1 * self.eval_fps))

                start = max(0, min(start, num_frames))
                end = max(0, min(end, num_frames))
                if end <= start:
                    end = min(num_frames, start + 1)

                for fi in range(start, end):
                    clip_behavior_list[fi].add(label)
                segments.append({"start": start, "end": end, "label": label})

            self.gt_anno[clip_name] = clip_behavior_list
            self.gt_segments[clip_name] = segments

    def _create_label_encoding(self):
        """Create integer encoding for labels for faster comparisons."""
        all_labels = set()
        for clip_name, label_sets in self.gt_anno.items():
            for frame_labels in label_sets:
                all_labels.update(frame_labels)

        # Create bidirectional mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(all_labels)

        # Convert GT annotations to binary matrices [num_frames, num_classes]
        self.gt_anno_encoded = {}
        for clip_name, label_sets in self.gt_anno.items():
            num_frames = len(label_sets)
            encoded = np.zeros((num_frames, self.num_classes), dtype=np.int8)
            for fi, frame_labels in enumerate(label_sets):
                for label in frame_labels:
                    encoded[fi, self.label_to_idx[label]] = 1
            self.gt_anno_encoded[clip_name] = encoded

    def _process_single_video_prediction(self, video_clip, predictions):
        """Process predictions for a single video (for parallel execution)."""
        behavior_clip = []
        for prediction_clip in predictions:
            if prediction_clip["score"] < 0.001:
                continue
            behavior_clip.append((prediction_clip["segment"], prediction_clip["label"], prediction_clip["score"]))

        # Sort the behavior clips by their start time
        behavior_clip.sort(key=lambda x: x[0][0])

        num_frames = self.video_frames.get(video_clip)
        if num_frames is None:
            num_frames = 0

        if num_frames == 0:
            return video_clip, [set() for _ in range(num_frames)], {}

        # OPTIMIZATION: Vectorized frame assignment
        # Convert segments to frame indices. Two paths:
        #   - PTS-aware: searchsorted into the clip-relative PTS array.
        #     Correct for CFR and VFR alike.
        #   - Legacy CFR fallback: `t * eval_fps` rounding.
        clip_pts = self.clip_pts.get(video_clip)
        segments_array = []
        labels_list = []
        scores_list = []

        for segment, label, score in behavior_clip:
            if clip_pts is not None:
                start_frame = int(np.searchsorted(clip_pts, float(segment[0]), side="left"))
                end_frame = int(np.searchsorted(clip_pts, float(segment[1]), side="right"))
            else:
                start_frame = int(segment[0] * self.eval_fps)
                end_frame = int(segment[1] * self.eval_fps)
            # Clip to valid range
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames))

            if end_frame > start_frame:
                segments_array.append((start_frame, end_frame, label, score))
                labels_list.append(label)
                scores_list.append(score)

        # Get unique labels in this video
        unique_labels_in_video = list(set(labels_list))

        # If no valid labels/segments, return empty prediction
        if len(unique_labels_in_video) == 0:
            return video_clip, [set() for _ in range(num_frames)], {i: {} for i in range(num_frames)}

        # Initialize score matrix: [num_frames, num_labels]
        label_idx_map = {label: i for i, label in enumerate(unique_labels_in_video)}
        score_matrix = np.zeros((num_frames, len(unique_labels_in_video)), dtype=np.float32)

        # Vectorized assignment of scores to frames
        for start_frame, end_frame, label, score in segments_array:
            label_idx = label_idx_map[label]
            # Take max score if multiple segments overlap
            score_matrix[start_frame:end_frame, label_idx] = np.maximum(
                score_matrix[start_frame:end_frame, label_idx], score
            )

        # Multi-label: keep all labels with score > 0 per frame
        whole_clip_label = []
        for frame_idx in range(num_frames):
            frame_labels = set()
            for label_idx, label in enumerate(unique_labels_in_video):
                if score_matrix[frame_idx, label_idx] > 0:
                    frame_labels.add(label)
            whole_clip_label.append(frame_labels)

        # Convert score matrix to frame_scores format (dict of dicts)
        frame_scores = {}
        for frame_idx in range(num_frames):
            frame_dict = {}
            for label_idx, label in enumerate(unique_labels_in_video):
                score = score_matrix[frame_idx, label_idx]
                if score > 0:
                    frame_dict[label] = float(score)
            frame_scores[frame_idx] = frame_dict

        return video_clip, whole_clip_label, frame_scores

    def _import_prediction(self, prediction_data):
        video_labels = {}
        video_frame_scores = {}

        video_clips = list(prediction_data["results"].keys())

        # Parallel processing of videos
        if self.thread > 1 and len(video_clips) > 1:
            with ThreadPoolExecutor(max_workers=self.thread) as executor:
                futures = {
                    executor.submit(
                        self._process_single_video_prediction,
                        video_clip,
                        prediction_data["results"][video_clip]
                    ): video_clip
                    for video_clip in video_clips
                }

                for future in as_completed(futures):
                    video_clip, whole_clip_label, frame_scores = future.result()
                    video_labels[video_clip] = whole_clip_label
                    video_frame_scores[video_clip] = frame_scores
        else:
            # Sequential processing
            for video_clip in video_clips:
                video_clip, whole_clip_label, frame_scores = self._process_single_video_prediction(
                    video_clip, prediction_data["results"][video_clip]
                )
                video_labels[video_clip] = whole_clip_label
                video_frame_scores[video_clip] = frame_scores

        self.pred_data = video_labels
        self.pred_frame_scores = video_frame_scores

    def compute_average_precision(self, scores, labels):
        """Compute Average Precision for a single class using COCO-style all-point interpolation.

        Args:
            scores: array of prediction scores for this class
            labels: binary array (1 if GT is this class, 0 otherwise)

        Returns:
            AP value
        """
        # Sort by score descending
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices]

        # Compute precision and recall at each threshold
        tp_cumsum = np.cumsum(sorted_labels)
        fp_cumsum = np.cumsum(1 - sorted_labels)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / np.sum(labels) if np.sum(labels) > 0 else np.zeros_like(tp_cumsum)

        # Add sentinel values at the beginning: precision=0 at recall=0
        recall = np.concatenate([[0], recall])
        precision = np.concatenate([[0], precision])

        # Make precision monotonically decreasing (from right to left)
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])

        # Compute area under the interpolated PR curve
        ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
        return ap

    def _build_frame_matrices(self):
        """Build aligned ``[N_frames, N_classes]`` score + GT-binary matrices
        over **all** frames of the videos common to GT and predictions.

        Background (empty-GT) frames are INCLUDED: their GT row is all-zero, so
        a prediction firing on a background frame counts as a false positive.
        That inclusion is exactly what makes the headline mAP and the threshold
        sweep penalize over-prediction (the high-recall / low-precision failure
        mode). A boolean ``nonempty`` mask is returned alongside so the legacy
        "exclude empty GT frames" mAP can still be reported as a diagnostic.

        Returns ``(scores, gt_binary, nonempty_mask, labels)`` and caches it.
        """
        if self._frame_matrices is not None:
            return self._frame_matrices

        common_videos = sorted(set(self.gt_anno.keys()) & set(self.pred_frame_scores.keys()))
        labels = sorted(self.label_to_idx.keys())
        col = {label: i for i, label in enumerate(labels)}

        total_frames = sum(self.video_frames.get(c, 0) for c in common_videos)
        scores = np.zeros((total_frames, len(labels)), dtype=np.float32)
        gt_binary = np.zeros((total_frames, len(labels)), dtype=np.int8)
        nonempty = np.zeros(total_frames, dtype=bool)

        row = 0
        for clip_name in common_videos:
            num_frames = self.video_frames[clip_name]
            gt_labels_array = self.gt_anno[clip_name]
            frame_scores_dict = self.pred_frame_scores[clip_name]
            for frame_idx in range(num_frames):
                frame_gt_labels = gt_labels_array[frame_idx]
                if frame_gt_labels:
                    nonempty[row] = True
                    for label in frame_gt_labels:
                        if label in col:
                            gt_binary[row, col[label]] = 1
                for label, score in frame_scores_dict.get(frame_idx, {}).items():
                    if label in col:
                        scores[row, col[label]] = score
                row += 1

        self._frame_matrices = (scores, gt_binary, nonempty, labels)
        return self._frame_matrices

    def compute_frame_based_mAP(self):
        """Compute frame-level mean Average Precision (background-aware).

        ``mAP`` is computed over ALL frames — background (empty-GT) frames act as
        negatives for every class, so a model that over-fires is penalized rather
        than rewarded. Supports multi-label: a frame can carry several GT labels
        at once.
        """
        scores, gt_binary, _nonempty, labels = self._build_frame_matrices()

        if scores.shape[0] == 0 or len(labels) == 0:
            return {"mAP": 0.0, "per_class_AP": {}, "evaluated_labels": []}

        per_class_AP = {}
        aps = []
        for i, label in enumerate(labels):
            col_scores = scores[:, i]
            col_gt = gt_binary[:, i]
            if np.sum(col_gt) > 0:
                ap = self.compute_average_precision(col_scores, col_gt)
                per_class_AP[label] = ap
                aps.append(ap)
            else:
                per_class_AP[label] = 0.0

        mAP = float(np.mean(aps)) if aps else 0.0

        return {
            "mAP": mAP,
            "per_class_AP": per_class_AP,
            "evaluated_labels": labels,
        }

    def compute_threshold_sweep(self, thresholds=None):
        """Sweep score thresholds and recommend F1-optimal cutoffs.

        Computes per-class precision/recall/F1 over ALL frames (background
        included, so false positives count) at each threshold, then reports:
          - ``per_class``: the threshold maximizing each class's F1, and
          - ``global``: the single threshold maximizing mean per-class F1.

        Tuned on whatever subset this evaluator runs over (validation, by
        convention). Returns a dict ready to drop into ``metrics.json``.
        """
        if thresholds is None:
            thresholds = [round(0.05 * k, 2) for k in range(1, 20)]  # 0.05 .. 0.95
        thresholds = [float(t) for t in thresholds]

        scores, gt_binary, _nonempty, labels = self._build_frame_matrices()

        per_class = {}
        f1_curves = []
        for i, label in enumerate(labels):
            col_scores = scores[:, i]
            gt_pos = gt_binary[:, i].astype(bool)
            n_pos = int(np.sum(gt_pos))
            best = {"threshold": thresholds[0], "f1": 0.0, "precision": 0.0, "recall": 0.0}
            curve = []
            for t in thresholds:
                pred = col_scores >= t
                tp = int(np.sum(pred & gt_pos))
                fp = int(np.sum(pred & ~gt_pos))
                fn = n_pos - tp
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                curve.append(f1)
                if f1 > best["f1"]:
                    best = {"threshold": t, "f1": f1, "precision": precision, "recall": recall}
            per_class[label] = best
            f1_curves.append(curve)

        if f1_curves:
            mean_f1 = np.mean(np.asarray(f1_curves), axis=0)
            gi = int(np.argmax(mean_f1))
            global_best = {"threshold": thresholds[gi], "f1": float(mean_f1[gi])}
        else:
            global_best = {"threshold": thresholds[0], "f1": 0.0}

        return {"per_class": per_class, "global": global_best}

    def compute_frame_based_precision(self, score_threshold=0.0):
        """Compute frame-based precision, recall, and F1 at a score threshold.

        A frame is a positive prediction for a class when that class's score is
        ``>= score_threshold`` (over all frames, background included). With
        ``score_threshold <= 0`` any non-zero prediction counts — the historical
        behavior that reported near-100% recall regardless of confidence.
        Supports multi-label: each class is scored independently.
        """
        scores, gt_binary, _nonempty, labels = self._build_frame_matrices()
        num_labels = len(labels)
        precision_list = np.zeros(num_labels, dtype=np.float64)
        recall_list = np.zeros(num_labels, dtype=np.float64)
        f1_list = np.zeros(num_labels, dtype=np.float64)

        if scores.shape[0] == 0 or num_labels == 0:
            return {
                "labels": labels,
                "precision": precision_list,
                "recall": recall_list,
                "f1": f1_list,
            }

        for i in range(num_labels):
            col_scores = scores[:, i]
            gt_col = gt_binary[:, i].astype(bool)
            if score_threshold > 0:
                pred_col = col_scores >= score_threshold
            else:
                pred_col = col_scores > 0

            tp = np.sum(gt_col & pred_col)
            fp = np.sum((~gt_col) & pred_col)
            fn = np.sum(gt_col & (~pred_col))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precision_list[i] = precision
            recall_list[i] = recall
            f1_list[i] = f1

        return {
            "labels": labels,
            "precision": precision_list,
            "recall": recall_list,
            "f1": f1_list,
        }

    def evaluate(self):
        # Background-aware mAP (the only mAP reported).
        mAP_metrics = self.compute_frame_based_mAP()
        self.mAP = mAP_metrics["mAP"]
        self.per_class_AP = mAP_metrics["per_class_AP"]
        self.mAP_labels = mAP_metrics["evaluated_labels"]

        # Threshold recommendation (per-class F1-optimal + global).
        self.threshold_sweep = self.compute_threshold_sweep()
        self.recommended_threshold = self.threshold_sweep["global"]["threshold"]

        # Precision/recall/F1 reported at the evaluation threshold: the value
        # configured on the evaluator, else the F1-optimal global threshold.
        if self.score_threshold is not None:
            self.eval_threshold = self.score_threshold
        else:
            self.eval_threshold = self.recommended_threshold
        metrics = self.compute_frame_based_precision(score_threshold=self.eval_threshold)
        self.precision = metrics["precision"]
        self.recall = metrics["recall"]
        self.f1_score = metrics["f1"]
        self.unique_labels = metrics["labels"]

        metric_dict = {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "mAP": self.mAP,
            "per_class_AP": self.per_class_AP,
            "eval_threshold": self.eval_threshold,
            "recommended_thresholds": {
                "global": self.threshold_sweep["global"],
                "per_class": self.threshold_sweep["per_class"],
            },
        }

        return metric_dict

    def logging(self, logger=None):
        """Log evaluation results."""
        if logger is None:
            pprint = print
        else:
            pprint = logger.info

        pprint(f"Loaded annotations from {self.subset} subset.")
        pprint(f"Number of ground truth entries: {len(self.gt_anno.keys())}")
        pprint(f"Number of predictions: {len(self.pred_data.keys())}")
        pprint(f"GT fps: {self.gt_fps}, eval fps: {self.eval_fps}")
        pprint(f"Unique labels: {self.unique_labels}")

        # Precision/recall/F1 are reported AT a concrete score threshold so they
        # are no longer the misleading ~0-threshold (recall≈100%) numbers.
        thr_src = "configured" if self.score_threshold is not None else "F1-optimal"
        pprint(f"\nPrecision/recall/F1 @ threshold {self.eval_threshold:.2f} ({thr_src}):")
        precision_str = [f"{p*100:.2f}%" for p in self.precision]
        recall_str = [f"{r*100:.2f}%" for r in self.recall]
        f1_str = [f"{f*100:.2f}%" for f in self.f1_score]
        pprint(f"Frame-based precision: {precision_str}")
        pprint(f"Frame-based recall: {recall_str}")
        pprint(f"Frame-based F1-score: {f1_str}")

        # mAP is background-aware (empty GT frames count as negatives), so
        # over-prediction is penalized rather than rewarded.
        pprint(f"\nFrame-based mAP (background-aware, all frames): {self.mAP*100:.2f}%")
        pprint(f"Per-class AP:")
        for label in self.mAP_labels:
            ap = self.per_class_AP.get(label, 0.0)
            pprint(f"  {label}: {ap*100:.2f}%")

        # Recommended confidence thresholds (tuned on this subset by F1).
        gthr = self.threshold_sweep["global"]
        pprint(f"\nRecommended threshold (global, max mean-F1): "
               f"{gthr['threshold']:.2f}  (mean F1 {gthr['f1']*100:.2f}%)")
        pprint(f"Recommended threshold per class (max F1):")
        for label in self.unique_labels:
            best = self.threshold_sweep["per_class"].get(label)
            if best is None:
                continue
            pprint(f"  {label}: thr={best['threshold']:.2f} "
                   f"(F1 {best['f1']*100:.2f}%, P {best['precision']*100:.2f}%, "
                   f"R {best['recall']*100:.2f}%)")
