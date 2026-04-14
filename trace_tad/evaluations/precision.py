import json
import numpy as np
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from .builder import EVALUATORS, remove_duplicate_annotations

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
        # Convert segments to frame indices
        segments_array = []
        labels_list = []
        scores_list = []

        for segment, label, score in behavior_clip:
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

    def compute_frame_based_mAP(self):
        """Compute frame-level mean Average Precision.

        For each frame, we have per-class scores from all overlapping predictions.
        Frames with empty GT (no labels) are excluded.
        Supports multi-label: a frame can have multiple GT labels simultaneously.
        """
        gt_video_clip_name = set(self.gt_anno.keys())
        pred_video_clip_name = set(self.pred_frame_scores.keys())
        common_videos = gt_video_clip_name & pred_video_clip_name

        if len(common_videos) == 0:
            return {"mAP": 0.0, "per_class_AP": {}, "evaluated_labels": []}

        # Collect all non-empty GT labels
        all_labels = set()
        for clip_name in common_videos:
            for frame_labels in self.gt_anno[clip_name]:
                all_labels.update(frame_labels)

        all_labels = sorted(list(all_labels))

        if len(all_labels) == 0:
            return {"mAP": 0.0, "per_class_AP": {}, "evaluated_labels": []}

        # Pre-count total valid frames (frames with at least one GT label)
        total_valid_frames = 0
        for clip_name in common_videos:
            for frame_labels in self.gt_anno[clip_name]:
                if len(frame_labels) > 0:
                    total_valid_frames += 1

        if total_valid_frames == 0:
            return {"mAP": 0.0, "per_class_AP": {}, "evaluated_labels": []}

        num_labels = len(all_labels)
        label_to_col = {label: i for i, label in enumerate(all_labels)}

        # Pre-allocate: [total_valid_frames, num_labels]
        all_scores = np.zeros((total_valid_frames, num_labels), dtype=np.float32)
        all_gt_binary = np.zeros((total_valid_frames, num_labels), dtype=np.int8)

        # Fill arrays
        current_row = 0
        for clip_name in common_videos:
            num_frames = self.video_frames[clip_name]
            gt_labels_array = self.gt_anno[clip_name]
            frame_scores_dict = self.pred_frame_scores[clip_name]

            for frame_idx in range(num_frames):
                frame_gt_labels = gt_labels_array[frame_idx]

                # Skip frames with no GT labels
                if len(frame_gt_labels) == 0:
                    continue

                frame_score_dict = frame_scores_dict.get(frame_idx, {})

                # Fill scores and multi-label GT binary
                for label in all_labels:
                    col_idx = label_to_col[label]
                    all_scores[current_row, col_idx] = frame_score_dict.get(label, 0.0)
                    all_gt_binary[current_row, col_idx] = 1 if label in frame_gt_labels else 0

                current_row += 1

        # Compute AP for each class
        per_class_AP = {}
        valid_aps = []

        for label_idx, label in enumerate(all_labels):
            scores = all_scores[:, label_idx]
            labels = all_gt_binary[:, label_idx]

            if np.sum(labels) > 0:
                ap = self.compute_average_precision(scores, labels)
                per_class_AP[label] = ap
                valid_aps.append(ap)
            else:
                per_class_AP[label] = 0.0

        mAP = np.mean(valid_aps) if len(valid_aps) > 0 else 0.0

        return {
            "mAP": mAP,
            "per_class_AP": per_class_AP,
            "evaluated_labels": all_labels
        }

    def compute_frame_based_precision(self):
        """Compute frame-based precision, recall, and F1.

        Supports multi-label: both GT and predictions can have multiple labels per frame.
        Metrics are computed per-class using binary matrices.
        """
        gt_video_clip_name = set(self.gt_anno.keys())
        pred_video_clip_name = set(self.pred_data.keys())

        common_videos = gt_video_clip_name & pred_video_clip_name

        if len(common_videos) == 0 or self.num_classes == 0:
            return {
                "labels": [],
                "precision": np.zeros(0, dtype=np.float64),
                "recall": np.zeros(0, dtype=np.float64),
                "f1": np.zeros(0, dtype=np.float64),
            }

        if len(common_videos) < len(gt_video_clip_name):
            missing_in_pred = gt_video_clip_name - pred_video_clip_name
            print(f"Warning: {len(missing_in_pred)} GT videos not found in predictions")

        # Build GT and prediction binary matrices [total_frames, num_classes]
        all_gt_rows = []
        all_pred_rows = []

        for clip_name in common_videos:
            # GT: already encoded as binary matrix [num_frames, num_classes]
            all_gt_rows.append(self.gt_anno_encoded[clip_name])

            # Predictions: convert set-of-labels per frame to binary matrix
            pred_label_sets = self.pred_data[clip_name]
            num_frames = len(pred_label_sets)
            pred_encoded = np.zeros((num_frames, self.num_classes), dtype=np.int8)
            for fi, frame_labels in enumerate(pred_label_sets):
                for label in frame_labels:
                    if label in self.label_to_idx:
                        pred_encoded[fi, self.label_to_idx[label]] = 1
            all_pred_rows.append(pred_encoded)

        all_gt = np.concatenate(all_gt_rows, axis=0)    # [N, C]
        all_pred = np.concatenate(all_pred_rows, axis=0)  # [N, C]

        unique_labels = sorted(self.label_to_idx.keys())
        num_labels = len(unique_labels)
        precision_list = np.zeros(num_labels, dtype=np.float64)
        recall_list = np.zeros(num_labels, dtype=np.float64)
        f1_list = np.zeros(num_labels, dtype=np.float64)

        for i, label in enumerate(unique_labels):
            col = self.label_to_idx[label]
            gt_col = all_gt[:, col]
            pred_col = all_pred[:, col]

            tp = np.sum(gt_col & pred_col)
            fp = np.sum((~gt_col.astype(bool)) & pred_col.astype(bool))
            fn = np.sum(gt_col.astype(bool) & (~pred_col.astype(bool)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precision_list[i] = precision
            recall_list[i] = recall
            f1_list[i] = f1

        return {
            "labels": unique_labels,
            "precision": precision_list,
            "recall": recall_list,
            "f1": f1_list,
        }

    def evaluate(self):
        # compute frame-based metrics
        metrics = self.compute_frame_based_precision()

        self.precision = metrics["precision"]
        self.recall = metrics["recall"]
        self.f1_score = metrics["f1"]
        self.unique_labels = metrics["labels"]

        # compute frame-based mAP
        mAP_metrics = self.compute_frame_based_mAP()
        self.mAP = mAP_metrics["mAP"]
        self.per_class_AP = mAP_metrics["per_class_AP"]
        self.mAP_labels = mAP_metrics["evaluated_labels"]

        metric_dict = {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "mAP": self.mAP,
            "per_class_AP": self.per_class_AP,
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

        # Format arrays to two decimal places and append percent sign
        precision_str = [f"{p*100:.2f}%" for p in self.precision]
        recall_str = [f"{r*100:.2f}%" for r in self.recall]
        f1_str = [f"{f*100:.2f}%" for f in self.f1_score]
        # Print formatted arrays
        pprint(f"Frame-based precision: {precision_str}")
        pprint(f"Frame-based recall: {recall_str}")
        pprint(f"Frame-based F1-score: {f1_str}")

        # Print mAP results
        pprint(f"\nFrame-based mAP (excluding empty GT frames): {self.mAP*100:.2f}%")
        pprint(f"Per-class AP:")
        for label in self.mAP_labels:
            ap = self.per_class_AP.get(label, 0.0)
            pprint(f"  {label}: {ap*100:.2f}%")
