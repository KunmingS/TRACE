import json
import os
import numpy as np
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from .builder import EVALUATORS, remove_duplicate_annotations


def _average_precision_sklearn_style(scores, labels):
    """Average precision matching sklearn.metrics.average_precision_score.

    sklearn groups tied scores before integrating the precision-recall curve.
    Keeping that behavior matters for JSON-rounded detector scores, where many
    frames can share identical probabilities.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8).astype(bool)
    if scores.size == 0 or labels.sum() == 0:
        return 0.0

    order = np.argsort(scores, kind="mergesort")[::-1]
    scores = scores[order]
    labels = labels[order]

    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, labels.size - 1]

    tps = np.cumsum(labels)[threshold_idxs].astype(np.float64)
    fps = (1 + threshold_idxs - tps).astype(np.float64)
    precision = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)
    recall = tps / tps[-1]

    prev_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev_recall) * precision))


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
        prediction_min_score=0.001,
        map_frame_filter="nonempty_gt",
        ap_mode="interpolated",
        score_threshold=None,
        per_class_thresholds=None,
        threshold_mode="global",
        threshold_grid=None,
        threshold_source=None,
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
        self.prediction_min_score = float(prediction_min_score)
        self.map_frame_filter = map_frame_filter
        self.ap_mode = ap_mode
        if self.map_frame_filter not in ("nonempty_gt", "all"):
            raise ValueError(
                "Precision map_frame_filter must be 'nonempty_gt' or 'all', "
                f"got {self.map_frame_filter!r}"
            )
        if self.ap_mode not in ("interpolated", "sklearn"):
            raise ValueError(f"Unsupported Precision ap_mode: {self.ap_mode!r}")

        # --- Operating point for the reported precision/recall/F1 ------------
        # mAP is threshold-free, but precision/recall/F1 are not. TRACE emits
        # dense proposals, so "any score > 0 is a detection" marks essentially
        # every frame positive: recall pins to 100% and precision collapses to
        # each class's frame prevalence, telling you nothing about the model.
        # P/R/F1 are therefore always reported AT a concrete cutoff:
        #   score_threshold       pinned value, used verbatim
        #   per_class_thresholds  {label: cutoff}, used when mode="per_class"
        #   neither               tuned on THIS subset by maximizing F1
        # threshold_grid restricts the search to fixed values (default: the
        # observed scores, i.e. the exact optimum). threshold_source is a
        # provenance string echoed into the log and metrics.json.
        self.score_threshold = None if score_threshold is None else float(score_threshold)
        self.per_class_thresholds = dict(per_class_thresholds) if per_class_thresholds else None
        self.threshold_mode = threshold_mode
        self.threshold_grid = None if threshold_grid is None else [float(t) for t in threshold_grid]
        self.threshold_source = threshold_source
        if self.threshold_mode not in ("global", "per_class"):
            raise ValueError(
                "Precision threshold_mode must be 'global' or 'per_class', "
                f"got {self.threshold_mode!r}"
            )
        # Cache for the aligned per-frame score/GT matrices (built lazily once).
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
            if prediction_clip["score"] < self.prediction_min_score:
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
        if self.ap_mode == "sklearn":
            return _average_precision_sklearn_style(scores, labels)

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
        """Build the aligned ``[N_frames, N_classes]`` score and GT matrices.

        Every frame of every video common to GT and predictions gets one row, in
        a deterministic (sorted-by-clip) order. Empty-GT frames are kept and a
        boolean ``nonempty`` mask is returned alongside, so both mAP flavors and
        the threshold sweep can be derived from a single pass -- the sweep needs
        background frames (that is where the false positives are), while
        ``map_frame_filter='nonempty_gt'`` masks them out.

        Returns ``(scores, gt_binary, nonempty, labels)``; result is cached.
        """
        if self._frame_matrices is not None:
            return self._frame_matrices

        common_videos = sorted(set(self.gt_anno.keys()) & set(self.pred_frame_scores.keys()))
        missing_in_pred = set(self.gt_anno.keys()) - set(self.pred_frame_scores.keys())
        if missing_in_pred:
            print(f"Warning: {len(missing_in_pred)} GT videos not found in predictions")

        labels = set()
        for clip_name in common_videos:
            for frame_labels in self.gt_anno[clip_name]:
                labels.update(frame_labels)
        labels = sorted(labels)
        col = {label: i for i, label in enumerate(labels)}

        total_frames = sum(self.video_frames[c] for c in common_videos)
        scores = np.zeros((total_frames, len(labels)), dtype=np.float32)
        gt_binary = np.zeros((total_frames, len(labels)), dtype=np.int8)
        nonempty = np.zeros(total_frames, dtype=bool)

        row = 0
        for clip_name in common_videos:
            gt_labels_array = self.gt_anno[clip_name]
            frame_scores_dict = self.pred_frame_scores[clip_name]

            for frame_idx in range(self.video_frames[clip_name]):
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
        """Compute frame-level mean Average Precision.

        For each frame, we have per-class scores from all overlapping predictions.
        By default, frames with empty GT (no labels) are excluded to preserve
        TRACE's historical Precision metric. Set ``map_frame_filter='all'`` and
        ``ap_mode='sklearn'`` to match FERAL's CalMS21-style mAP: all frames are
        used as one-vs-rest examples and only behavior classes are averaged.
        Supports multi-label: a frame can have multiple GT labels simultaneously.
        """
        all_scores, all_gt_binary, nonempty, all_labels = self._build_frame_matrices()

        if self.map_frame_filter == "nonempty_gt":
            all_scores = all_scores[nonempty]
            all_gt_binary = all_gt_binary[nonempty]

        if len(all_labels) == 0 or all_scores.shape[0] == 0:
            return {"mAP": 0.0, "per_class_AP": {}, "evaluated_labels": []}

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

    @staticmethod
    def _build_class_curve(scores_col, gt_pos):
        """Precompute what is needed to score ANY threshold for one class.

        Ascending sorted positive scores plus suffix true-positive counts turn
        each threshold query into one ``searchsorted`` instead of a fresh O(N)
        mask, which is what makes an exhaustive sweep affordable on evaluations
        with hundreds of thousands of frames.

        Frames scored 0 can never be predicted positive, so they are dropped from
        the sorted arrays; GT positives among them stay counted in ``n_pos`` and
        therefore still register as false negatives.
        """
        pos_mask = scores_col > 0
        s = scores_col[pos_mask].astype(np.float64)
        y = gt_pos[pos_mask].astype(np.float64)
        order = np.argsort(s, kind="mergesort")
        s = s[order]
        y = y[order]
        # tp_suffix[i] = positives among s[i:]; the trailing 0 covers "keep none".
        tp_suffix = np.concatenate([np.cumsum(y[::-1])[::-1], [0.0]])
        return {"s": s, "tp_suffix": tp_suffix, "n_pos": float(np.count_nonzero(gt_pos))}

    @staticmethod
    def _prf_at(curve, thresholds):
        """Precision/recall/F1 for one class at each threshold (``score >= t``)."""
        s, tp_suffix, n_pos = curve["s"], curve["tp_suffix"], curve["n_pos"]
        t = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
        first = np.searchsorted(s, t, side="left")  # first index with score >= t
        tp = tp_suffix[first]
        fp = (s.size - first) - tp
        denom_p = tp + fp
        precision = np.where(denom_p > 0, tp / np.maximum(denom_p, 1e-12), 0.0)
        recall = np.full_like(precision, 0.0) if n_pos <= 0 else tp / n_pos
        denom_f = precision + recall
        f1 = np.where(denom_f > 0, 2 * precision * recall / np.maximum(denom_f, 1e-12), 0.0)
        return precision, recall, f1

    def _global_threshold_candidates(self, scores):
        """Candidate cutoffs for the single shared threshold: score quantiles.

        A fixed 0.05-step grid can miss the useful range entirely -- a detector's
        scores often crowd into a narrow band -- so the candidates follow the
        distribution the model actually produced. Capped at 512 values to keep
        the search cheap.
        """
        pooled = scores[scores > 0].astype(np.float64)
        if pooled.size == 0:
            return np.array([1.0])
        return np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, 512)))

    def compute_threshold_tuning(self):
        """Find the F1-optimal score thresholds on this subset.

        Reports the per-class optimum and the single global cutoff maximizing the
        mean per-class F1 (the number quoted alongside P/R/F1). Per-class optima
        are exact when ``threshold_grid`` is unset: F1 only changes at observed
        scores, so sweeping the distinct scores finds the true argmax with no
        resolution parameter to pick.

        Tuning and reporting on the same subset is optimistic, so this is a
        *recommendation*: export it from the validation run and pin it via
        ``score_threshold`` when scoring the held-out split.
        """
        scores, gt_binary, _nonempty, labels = self._build_frame_matrices()
        curves = [
            self._build_class_curve(scores[:, i], gt_binary[:, i].astype(bool))
            for i in range(len(labels))
        ]

        no_signal = {"threshold": 1.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        per_class = {}
        for label, curve in zip(labels, curves):
            if curve["n_pos"] <= 0 or curve["s"].size == 0:
                per_class[label] = dict(no_signal)
                continue
            cand = (np.unique(curve["s"]) if self.threshold_grid is None
                    else np.asarray(self.threshold_grid, dtype=np.float64))
            precision, recall, f1 = self._prf_at(curve, cand)
            k = int(np.argmax(f1))
            per_class[label] = {
                "threshold": float(cand[k]),
                "f1": float(f1[k]),
                "precision": float(precision[k]),
                "recall": float(recall[k]),
            }

        if not labels:
            return {"global": dict(no_signal, mean_f1=0.0), "per_class": per_class}

        if self.threshold_grid is not None:
            cand = np.asarray(self.threshold_grid, dtype=np.float64)
        else:
            # Quantile grid, plus each class's own optimum as a candidate.
            cand = np.unique(np.concatenate([
                self._global_threshold_candidates(scores),
                np.array([per_class[l]["threshold"] for l in labels], dtype=np.float64),
            ]))

        f1_matrix = np.vstack([self._prf_at(c, cand)[2] for c in curves])
        mean_f1 = f1_matrix.mean(axis=0)
        k = int(np.argmax(mean_f1))
        t_star = float(cand[k])
        stats = [self._prf_at(c, t_star) for c in curves]
        global_best = {
            "threshold": t_star,
            "mean_f1": float(mean_f1[k]),
            "mean_precision": float(np.mean([p[0] for p, _r, _f in stats])),
            "mean_recall": float(np.mean([r[0] for _p, r, _f in stats])),
        }

        return {"global": global_best, "per_class": per_class}

    def compute_frame_based_precision(self, thresholds=None):
        """Compute frame-based precision, recall, and F1 at a score threshold.

        ``thresholds`` maps label -> cutoff; a frame counts as a positive
        prediction for a class when that class's score is ``>= cutoff``. A
        missing or ``None`` entry falls back to "any non-zero score counts",
        which is the historical behavior that pinned recall at ~100%.

        Metrics run over ALL frames, so predictions firing on background frames
        count as false positives. Supports multi-label: both GT and predictions
        can carry several labels per frame, and each class is scored separately.
        """
        scores, gt_binary, _nonempty, labels = self._build_frame_matrices()
        thresholds = thresholds or {}

        num_labels = len(labels)
        precision_list = np.zeros(num_labels, dtype=np.float64)
        recall_list = np.zeros(num_labels, dtype=np.float64)
        f1_list = np.zeros(num_labels, dtype=np.float64)

        if num_labels == 0 or scores.shape[0] == 0:
            return {
                "labels": labels,
                "precision": precision_list,
                "recall": recall_list,
                "f1": f1_list,
            }

        for i, label in enumerate(labels):
            scores_col = scores[:, i]
            gt_col = gt_binary[:, i].astype(bool)
            threshold = thresholds.get(label)

            if threshold is None:
                pred_col = scores_col > 0
            else:
                pred_col = scores_col >= float(threshold)

            tp = np.count_nonzero(gt_col & pred_col)
            fp = np.count_nonzero((~gt_col) & pred_col)
            fn = np.count_nonzero(gt_col & (~pred_col))

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

    def _resolve_eval_thresholds(self, labels, tuning):
        """Pick the cutoff(s) the reported P/R/F1 are computed at.

        Precedence: values pinned on the evaluator win, and whatever is not
        pinned falls back to this subset's F1-optimal value. Returns
        ``(thresholds_by_label, mode)`` where ``mode`` names the provenance so
        the log can never leave it ambiguous.
        """
        if self.threshold_mode == "per_class":
            pinned = self.per_class_thresholds or {}
            fallback = self.score_threshold
            thresholds = {}
            any_pinned = False
            any_tuned = False
            for label in labels:
                value = pinned.get(label)
                if value is None:
                    value = fallback
                else:
                    any_pinned = True
                if value is None:
                    value = tuning["per_class"].get(label, {}).get("threshold")
                    any_tuned = True
                thresholds[label] = None if value is None else float(value)
            if any_pinned and any_tuned:
                mode = "per-class, partly configured / partly tuned on this subset"
            elif any_pinned:
                mode = "per-class, configured"
            else:
                mode = "per-class, tuned on this subset"
            return thresholds, mode

        if self.score_threshold is not None:
            value, mode = self.score_threshold, "configured"
        else:
            value, mode = tuning["global"]["threshold"], "tuned on this subset"
        return {label: float(value) for label in labels}, mode

    def evaluate(self):
        # compute frame-based mAP first -- it is threshold-free, so nothing below
        # can move it
        mAP_metrics = self.compute_frame_based_mAP()
        self.mAP = mAP_metrics["mAP"]
        self.per_class_AP = mAP_metrics["per_class_AP"]
        self.mAP_labels = mAP_metrics["evaluated_labels"]

        # always tune, even when a threshold is pinned: the recommendation is
        # what tells you whether the pinned operating point is a sane one
        self.threshold_tuning = self.compute_threshold_tuning()

        _scores, _gt, _nonempty, labels = self._build_frame_matrices()
        self.eval_thresholds, self.eval_threshold_mode = self._resolve_eval_thresholds(
            labels, self.threshold_tuning
        )
        distinct = {v for v in self.eval_thresholds.values()}
        self.eval_threshold = distinct.pop() if len(distinct) == 1 else None

        # compute frame-based metrics at that operating point
        metrics = self.compute_frame_based_precision(self.eval_thresholds)
        self.precision = metrics["precision"]
        self.recall = metrics["recall"]
        self.f1_score = metrics["f1"]
        self.unique_labels = metrics["labels"]

        # the pre-threshold numbers, kept as a diagnostic: every non-zero score
        # counts as a detection, so recall saturates and precision degenerates to
        # each class's frame prevalence
        untuned = self.compute_frame_based_precision(None)

        metric_dict = {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "mAP": self.mAP,
            "per_class_AP": self.per_class_AP,
            "mAP_frame_filter": self.map_frame_filter,
            "AP_mode": self.ap_mode,
            "labels": self.unique_labels,
            "eval_threshold": self.eval_threshold,
            "eval_thresholds": self.eval_thresholds,
            "eval_threshold_mode": self.eval_threshold_mode,
            "eval_threshold_source": self.threshold_source,
            "threshold_tuning": self.threshold_tuning,
            "metrics_at_any_score": {
                "precision": untuned["precision"],
                "recall": untuned["recall"],
                "f1_score": untuned["f1"],
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

        # Precision/recall/F1 only mean something at a stated cutoff, so name it
        # on the same line as the numbers.
        if self.eval_threshold is not None:
            at = f"@ score >= {self.eval_threshold:.4f}"
        else:
            per = ", ".join(
                f"{label}={self.eval_thresholds[label]:.4f}"
                if self.eval_thresholds.get(label) is not None else f"{label}=any"
                for label in self.unique_labels
            )
            at = f"@ per-class score >= ({per})"
        pprint(f"Score threshold {at} [{self.eval_threshold_mode}]")
        if self.threshold_source:
            pprint(f"Threshold source: {self.threshold_source}")

        # Format arrays to two decimal places and append percent sign
        precision_str = [f"{p*100:.2f}%" for p in self.precision]
        recall_str = [f"{r*100:.2f}%" for r in self.recall]
        f1_str = [f"{f*100:.2f}%" for f in self.f1_score]
        # Print formatted arrays
        pprint(f"Frame-based precision: {precision_str}")
        pprint(f"Frame-based recall: {recall_str}")
        pprint(f"Frame-based F1-score: {f1_str}")

        if self.map_frame_filter == "all":
            map_desc = "all frames, excluding other/background from class average"
        else:
            map_desc = "excluding empty GT frames"
        pprint(f"\nFrame-based mAP ({map_desc}; AP={self.ap_mode}): {self.mAP*100:.2f}%")
        pprint(f"Per-class AP:")
        for label in self.mAP_labels:
            ap = self.per_class_AP.get(label, 0.0)
            pprint(f"  {label}: {ap*100:.2f}%")

        # F1-optimal cutoffs measured on THIS subset. Tuned and reported on the
        # same data, so they are an upper bound -- export them from validation and
        # pin them via `score_threshold` for held-out reporting.
        tuning = self.threshold_tuning
        gbest = tuning["global"]
        pprint(f"\nF1-optimal thresholds on '{self.subset}':")
        pprint(f"  global: {gbest['threshold']:.4f} -> mean F1 {gbest['mean_f1']*100:.2f}% "
               f"(P {gbest.get('mean_precision', 0.0)*100:.2f}%, "
               f"R {gbest.get('mean_recall', 0.0)*100:.2f}%)")
        for label in self.unique_labels:
            best = tuning["per_class"].get(label)
            if best is None:
                continue
            pprint(f"  {label}: {best['threshold']:.4f} -> F1 {best['f1']*100:.2f}% "
                   f"(P {best['precision']*100:.2f}%, R {best['recall']*100:.2f}%)")
