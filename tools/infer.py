"""Standalone inference script for TRACE.

Runs temporal action detection on video files without requiring annotation JSONs.
Automatically probes videos for frame count, duration, and a per-frame PTS
table (cached as ``<video>.pts.npy`` next to the source — see
``pts-based-frame-mapping.md (archived)``). Before running the model, source videos
are split into small cached clips under the prediction work directory so
inference workers do not keep decoding long raw files.

Usage:
    python tools/infer.py configs/maev2b.py \
        --checkpoint runs/maev2b/checkpoint_best.pth \
        --input /path/to/videos \
        --class-map data/CALMS21/classmap.txt

    python tools/infer.py configs/maev2b.py \
        --checkpoint runs/maev2b/checkpoint_best.pth \
        --input /path/to/single_video.mp4 \
        --class-map data/CALMS21/classmap.txt \
        --output /path/to/prediction/folder
"""
import argparse
import csv
import json
import os
import shutil
import sys
import tempfile

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vtrace.config import Config, DictAction, num_classes_cfg
from vtrace.models import build_detector
from vtrace.datasets import build_dataset, build_dataloader
from vtrace.cores import eval_one_epoch
from vtrace.utils import set_seed, update_workdir, create_folder, setup_logger


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
VIDEO_VARIANT_ORDER = {"source": 0, "remux": 1, "h264": 2}


def _strip_known_video_extension(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    return name[:-len(ext)] if ext in VIDEO_EXTENSIONS else name


def _parse_video_file(path_or_name: str):
    """Return (canonical_stem, variant) using the PathPicker grouping rule."""
    name = os.path.basename(path_or_name)
    lower_name = name.lower()
    if lower_name.endswith(".remux.mp4"):
        return _strip_known_video_extension(name[:-len(".remux.mp4")]), "remux"
    if lower_name.endswith(".h264.mp4"):
        return _strip_known_video_extension(name[:-len(".h264.mp4")]), "h264"

    ext = os.path.splitext(name)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return None
    return name[:-len(ext)], "source"


def probe_video(filepath):
    """Probe a video and (re)build its PTS-table cache.

    Returns ``(num_frames, duration, fps, pts_path)``:

    - ``num_frames`` — encoded frame count read from the container index.
    - ``duration`` — PTS-derived: ``pts[-1] - pts[0] + 1/avg_fps``. Matches
      the convention used by ``vtrace.data_prep`` so train, eval, and
      inference all agree on clip length.
    - ``fps`` — average fps from decord (display-only; no time ↔ frame
      math goes through it once the PTS table is available). Falls back
      to PTS-span estimation if the container reports an invalid value.
    - ``pts_path`` — absolute path of the cached ``<video>.pts.npy`` if
      it was successfully written, else ``None``.

    See ``pts-based-frame-mapping.md (archived)`` for the design and why this
    matters for VFR webcam recordings.
    """
    from decord import VideoReader
    from vtrace.data_prep import _load_or_build_pts, _pts_cache_path

    pts = _load_or_build_pts(filepath)
    num_frames = len(pts)

    # Average fps purely for display + as a duration tail-correction. The
    # presentation-time arithmetic itself never divides by fps.
    fps = VideoReader(filepath, num_threads=1).get_avg_fps()
    if fps is None or fps <= 0:
        span = float(pts[-1] - pts[0]) if num_frames > 1 else 0.0
        fps = (num_frames - 1) / span if span > 0 else 30.0

    if num_frames > 1:
        duration = float(pts[-1] - pts[0]) + (1.0 / fps)
    else:
        duration = 0.0

    cache = _pts_cache_path(filepath)
    pts_path = os.path.abspath(cache) if os.path.isfile(cache) else None

    return num_frames, duration, fps, pts_path


def discover_videos(input_path):
    """Discover canonical video files from a path (file or directory).

    Annotation can create browser-ready copies next to the source, such as
    ``trial.mkv.remux.mp4`` or ``trial.mkv.h264.mp4``. Treat those as variants
    of the same source stem and keep one input, preferring the original.
    """
    videos = []
    if os.path.isfile(input_path):
        videos.append(input_path)
    elif os.path.isdir(input_path):
        by_stem = {}
        for index, fname in enumerate(sorted(os.listdir(input_path))):
            parsed = _parse_video_file(fname)
            if not parsed:
                continue
            stem, variant = parsed
            rank = VIDEO_VARIANT_ORDER[variant]
            path = os.path.join(input_path, fname)
            prev = by_stem.get(stem)
            if prev is None or rank < prev[0]:
                by_stem[stem] = (rank, index, path)
        videos = [
            path for _, _, path in sorted(
                by_stem.values(),
                key=lambda item: item[1],
            )
        ]
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not videos:
        raise FileNotFoundError(f"No video files found in {input_path}")
    return videos


def _proxy_workers_from_cfg(cfg, fallback=4):
    try:
        return int(cfg.solver.test.num_workers)
    except Exception:
        return fallback


def generate_pseudo_annotations(
    video_paths,
    logger,
    cache_dir=None,
    clip_frames=768,
    proxy_geometry=None,
    proxy_crf=None,
    proxy_workers=None,
):
    """Probe videos and create a temporary annotation JSON for inference.

    Windows are virtual: each entry records ``source_video`` /
    ``source_frame_offset`` / ``source_pts_table`` so predictions map back onto
    the source timeline exactly. When ``proxy_geometry`` is given, a downscaled
    full-length proxy of each video is built once and decoded from instead of
    the original.
    """
    from vtrace.data_prep import (
        PROXY_CRF,
        enumerate_virtual_clips,
        ensure_video_proxies,
    )

    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="trace_infer_cache_")
    if proxy_crf is None:
        proxy_crf = PROXY_CRF

    probed = []
    for vpath in video_paths:
        video_name = _video_stem(vpath)
        try:
            num_frames, duration, fps, pts_path = probe_video(vpath)
        except Exception as e:
            logger.warning(f"Skipping {video_name}: {e}")
            continue

        logger.info(
            f"{video_name}: {num_frames} frames, {duration:.1f}s, {fps:.1f} fps"
        )
        if not pts_path:
            # Worth saying out loud: without the sidecar every later run pays
            # the scan again.
            logger.warning(f"  {video_name}: PTS cache unavailable (read-only dir?)")
        probed.append((vpath, video_name))

    proxies = {}
    if proxy_geometry is not None and probed:
        proxies = ensure_video_proxies(
            [vpath for vpath, _ in probed],
            proxy_geometry,
            crf=proxy_crf,
            workers=proxy_workers,
            logger=logger,
        )

    database = {}
    for vpath, video_name in probed:
        try:
            clips = enumerate_virtual_clips(
                vpath, clip_frames=clip_frames, clip_stem=video_name
            )
        except Exception as e:
            logger.warning(f"Skipping {video_name}: could not enumerate clips: {e}")
            continue

        proxy_path = proxies.get(os.path.abspath(vpath))
        logger.debug(
            f"  {len(clips)} virtual clip(s)"
            + (" decoding from proxy" if proxy_path else " decoding from source")
        )
        for clip in clips:
            clip_key = clip["clip_name"]
            entry = {
                "subset": "validation",
                "frame": clip["frame"],
                "duration": clip["duration"],
                "annotations": [],
                "source_video": clip["source_video"],
                "source_frame_offset": clip["source_frame_offset"],
                "source_prediction_name": video_name,
                "source_start_seconds": clip["source_start_seconds"],
            }
            if "source_pts_table" in clip:
                entry["source_pts_table"] = clip["source_pts_table"]
            if proxy_path:
                entry["proxy_video"] = proxy_path
            database[clip_key] = entry

    if not database:
        raise RuntimeError("No videos could be probed successfully")

    return {"database": database}


def aggregate_clip_predictions(raw_predictions, annotation_database):
    """Merge cached-clip predictions back onto source-video timelines."""
    predictions = {}
    for entry in annotation_database.values():
        source_name = entry.get("source_prediction_name")
        if source_name:
            predictions.setdefault(source_name, [])

    for clip_name, detections in raw_predictions.items():
        entry = annotation_database.get(clip_name, {})
        source_name = entry.get("source_prediction_name", clip_name)
        offset = float(entry.get("source_start_seconds", 0.0))
        target = predictions.setdefault(source_name, [])
        for det in detections:
            adjusted = dict(det)
            segment = adjusted.get("segment", [0.0, 0.0])
            adjusted["segment"] = [
                round(float(segment[0]) + offset, 2),
                round(float(segment[1]) + offset, 2),
            ]
            target.append(adjusted)

    for dets in predictions.values():
        dets.sort(key=lambda det: (float(det.get("segment", [0.0, 0.0])[0]), -float(det.get("score", 0.0))))
    return predictions


# A dense head scores every frame, so its raw output is one detection per frame
# (~0.033 s at 30 fps). A reviewer wants bouts, so consecutive same-label frames
# are run-length merged first. The gap tolerance bridges a dropped frame or two
# inside one behaviour without joining genuinely separate bouts. Raw per-frame
# scores stay in result_detection.json for anyone scoring frame-level metrics.
_BOUT_GAP_SECONDS = 0.2


def _merge_detections(detections, max_gap=_BOUT_GAP_SECONDS):
    """Run-length merge same-label detections that touch into single bouts.

    Returns (label, start, end, score) with the bout's score being the highest
    of the frames it covers — the merged span is at least that confident
    somewhere, which is what a reviewer sorting by confidence wants.
    """
    bouts = []
    for label in sorted({det["label"] for det in detections}):
        spans = sorted(
            ((det["segment"][0], det["segment"][1], float(det.get("score", 0.0)))
             for det in detections if det["label"] == label),
            key=lambda span: span[0],
        )
        if not spans:
            continue
        start, end, score = spans[0]
        for seg_start, seg_end, seg_score in spans[1:]:
            if seg_start <= end + max_gap:
                end = max(end, seg_end)
                score = max(score, seg_score)
            else:
                bouts.append((label, start, end, score))
                start, end, score = seg_start, seg_end, seg_score
        bouts.append((label, start, end, score))
    return sorted(bouts, key=lambda bout: (bout[1], bout[0]))


def _truth_bouts(video_path):
    """(label, start, end) from the annotation CSV beside `video_path`, if any."""
    stem_path = os.path.splitext(video_path)[0] + ".csv"
    if not os.path.isfile(stem_path):
        return None
    from vtrace.data_prep import _csv_dict_reader

    bouts = []
    with open(stem_path, "r", encoding="utf-8") as f:
        for row in _csv_dict_reader(f):
            try:
                bouts.append((row["labelId"].strip(),
                              float(row["timestamp"]),
                              float(row["endTimestamp"])))
            except (KeyError, ValueError):
                continue
    return bouts


def _spans_to_mask(spans, grid):
    """Boolean array over `grid` (seconds), True inside any of `spans`."""
    import numpy as np

    mask = np.zeros(len(grid), dtype=bool)
    for start, end in spans:
        first = int(np.searchsorted(grid, start, side="left"))
        last = int(np.searchsorted(grid, end, side="right"))
        mask[first:last] = True
    return mask


def _f1(gt, pr):
    tp = int((gt & pr).sum())
    fp = int((~gt & pr).sum())
    fn = int((gt & ~pr).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return (precision, recall,
            2 * precision * recall / (precision + recall) if precision + recall else 0.0)


def tune_thresholds(detections, truth, duration, step=0.02):
    """Per-class cutoff that maximises frame-level F1 against `truth`.

    Swept on the raw per-frame detections and re-merged at every candidate,
    because the threshold is not only a filter: it decides which frames survive
    to be run-length merged, and so changes the bouts themselves. Filtering an
    already-merged bout list gives a different — and much worse — answer.

    Returns {label: (cutoff, precision, recall, f1)}.
    """
    import numpy as np

    if not truth or duration <= 0:
        return {}
    # 100 Hz is finer than any frame rate here, so the grid is not what limits
    # agreement.
    grid = np.arange(0.0, duration, 0.01)
    labels = sorted({label for label, _s, _e in truth}
                    | {det["label"] for det in detections})
    tuned = {}
    for label in labels:
        gt = _spans_to_mask([(s, e) for lab, s, e in truth if lab == label], grid)
        mine = [det for det in detections if det["label"] == label]
        if not gt.any():
            # The annotator recorded none of this behaviour in this recording,
            # so every frame predicted is a false positive and the F1-optimal
            # cutoff is the one that predicts nothing. Without this the sweep
            # sees F1 = 0 everywhere, keeps the first candidate it tried, and
            # writes out the whole unfiltered class.
            tuned[label] = (1.0, 0.0, 0.0, 0.0)
            continue
        best = (1.0, 0.0, 0.0, -1.0)
        for cutoff in np.arange(0.0, 1.0, step):
            kept = [det for det in mine if float(det.get("score", 0.0)) >= cutoff]
            bouts = _merge_detections(kept)
            pr = _spans_to_mask([(b[1], b[2]) for b in bouts], grid)
            precision, recall, f1 = _f1(gt, pr)
            # `>=` rather than `>`: candidates ascend, so a tie keeps the
            # stricter cutoff — the same F1 with fewer false positives.
            if f1 >= best[3]:
                best = (float(cutoff), precision, recall, f1)
        tuned[label] = best
    return tuned


def filter_predictions(predictions, threshold):
    """Drop detections scoring below ``threshold``."""
    threshold = max(0.0, min(1.0, float(threshold)))
    return {
        video_name: [dict(det) for det in detections
                     if float(det.get("score", 0.0)) >= threshold]
        for video_name, detections in predictions.items()
    }


# One prediction file per video, beside it, replaced on re-run. Same three columns
# training reads plus a score, so a corrected prediction is already an annotation
# file rather than something to convert.
PREDICTION_SUFFIX = ".pred.csv"
PREDICTION_COLUMNS = ("labelId", "timestamp", "endTimestamp", "score", "predictionId")
# Beside the CSV, on request: every frame's score for every class. The CSV keeps
# only bouts above a cutoff, which is what a reviewer wants and exactly what a
# precision-recall curve cannot be drawn from.
FRAME_SCORES_SUFFIX = ".pred.scores.npz"


def prediction_path(video_path, output_dir=None):
    """Where `video_path`'s prediction file goes.

    Beside the video by default, so opening that folder in the annotator shows
    the video and its predictions together; `output_dir` overrides the folder
    while keeping the name.
    """
    stem = _video_stem(video_path)
    folder = output_dir or os.path.dirname(os.path.abspath(video_path))
    return os.path.join(folder, stem + PREDICTION_SUFFIX)


def frame_scores_path(video_path, output_dir=None):
    """Where `video_path`'s per-frame score file goes: beside its prediction CSV."""
    path = prediction_path(video_path, output_dir)
    return path[:-len(PREDICTION_SUFFIX)] + FRAME_SCORES_SUFFIX


def write_frame_scores(video_paths, predictions, class_map, output_dir=None, logger=None):
    """Write one `<video stem>.pred.scores.npz` per video; return name -> path.

    `scores[t, c]` is the highest detection score covering frame `t` for class
    `labels[c]`, on the video's own PTS grid (the `.pts.npy` beside it), 0 where
    nothing covered the frame. Detections map to frames exactly as the evaluator
    maps them, so frame-level AP computed from this file is the evaluator's AP.
    float16: a ten-minute video at 30 fps is ~100 KB for three classes.
    """
    import numpy as np
    from vtrace.data_prep import _load_or_build_pts

    labels = list(class_map)
    column = {label: index for index, label in enumerate(labels)}
    written = {}
    for video_path in video_paths:
        stem = _video_stem(video_path)
        pts = np.asarray(_load_or_build_pts(video_path), dtype=np.float64)
        matrix = np.zeros((len(pts), len(labels)), dtype=np.float32)
        for det in predictions.get(stem, []):
            col = column.get(det.get("label"))
            if col is None:
                continue
            start, end = det.get("segment", (0.0, 0.0))
            first = int(np.searchsorted(pts, float(start), side="left"))
            last = int(np.searchsorted(pts, float(end), side="right"))
            first = max(0, min(first, len(pts) - 1))
            last = max(first + 1, min(last, len(pts)))
            np.maximum(matrix[first:last, col], float(det.get("score", 0.0)),
                       out=matrix[first:last, col])
        path = frame_scores_path(video_path, output_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, scores=matrix.astype(np.float16),
                            labels=np.array(labels), video=os.path.basename(video_path))
        written[stem] = path
        if logger:
            logger.info(f"Frame scores: {path} ({len(pts)} frames x {len(labels)} classes)")
    return written


def write_prediction_files(
    video_paths, predictions, class_map, threshold, output_dir=None, logger=None,
    tuned=None,
):
    """Write one `<video stem>.pred.csv` per video; return name -> path.

    `tuned` maps a video stem to {label: (cutoff, precision, recall, f1)}. When
    present the per-class cutoff replaces the flat `threshold` for that video,
    and the header records which was used.
    """
    written = {}
    for video_path in video_paths:
        stem = _video_stem(video_path)
        cutoffs = (tuned or {}).get(stem) or {}
        raw = predictions.get(stem, [])
        if cutoffs:
            raw = [det for det in raw
                   if float(det.get("score", 0.0))
                   >= cutoffs.get(det["label"], (1.0,))[0]]
        bouts = [
            {
                "label": label,
                "segment": [round(start, 3), round(end, 3)],
                "score": round(score, 4),
            }
            for label, start, end, score in _merge_detections(raw)
        ] if raw else []

        path = prediction_path(video_path, output_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        meta = {
            "trace_prediction_version": 1,
            "video": os.path.basename(video_path),
            "class_map": list(class_map),
            "threshold": ({label: round(value[0], 3)
                           for label, value in sorted(cutoffs.items())}
                          if cutoffs else threshold),
        }
        if cutoffs:
            meta["threshold_tuned_on"] = "this video's own annotation CSV"
        with open(path, "w", encoding="utf-8", newline="") as f:
            # `#` lines are skipped by prep's reader and by the annotator's, so the
            # run's provenance rides along without becoming a column.
            f.write(f"# trace-meta: {json.dumps(meta)}\n")
            writer = csv.writer(f)
            writer.writerow(PREDICTION_COLUMNS)
            for index, bout in enumerate(bouts):
                writer.writerow([
                    bout["label"], bout["segment"][0], bout["segment"][1],
                    bout["score"], f"{stem}::{index}",
                ])
        written[stem] = path
        if logger:
            logger.info(f"Prediction CSV: {path} ({len(bouts)} bouts)")
    return written


def _shutdown_dataloader_workers(loader, logger):
    """Stop persistent DataLoader workers before CPU-heavy render/export work."""
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is None:
        return
    try:
        shutdown()
        loader._iterator = None
        logger.info("Stopped DataLoader workers before post-processing outputs.")
    except Exception as exc:
        logger.warning(f"Could not stop DataLoader workers cleanly: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on video files")
    parser.add_argument("config", metavar="CONFIG", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--input", type=str, required=True,
                        help="Input video file or directory of videos")
    parser.add_argument("--class-map", type=str, required=True,
                        help="Class map file (one class name per line)")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory for the prediction files "
                             "(default: beside each source video)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum score for a detection to reach the prediction file")
    parser.add_argument("--frame-scores", action="store_true",
                        help="Also write <stem>.pred.scores.npz beside each prediction "
                             "file: every frame's score for every class, for frame-level "
                             "metrics such as a precision-recall curve or AP.")
    parser.add_argument("--tune-threshold", action="store_true",
                        help="Choose the cutoff per class by maximising frame-level "
                             "F1 against the annotation CSV beside each video. "
                             "Tunes on the video it reports, so the result is an "
                             "upper bound, not a held-out estimate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--profile", action="store_true",
                        help="Enable inference profiling (CPU + GPU timing)")
    parser.add_argument("--auto-tune", action="store_true",
                        help="Auto-tune dataloader params based on system resources")
    parser.add_argument("--pairs", "--include-stems", dest="include_stems",
                        nargs="*", default=None,
                        help="Restrict inference to these video stems "
                             "(filename minus extension; .remux/.h264 copy "
                             "suffixes collapse to the source stem). Omit to "
                             "process every discovered video.")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction,
                        help="Override config settings (key=value pairs)")
    args = parser.parse_args()
    if args.threshold < 0.0 or args.threshold > 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def _video_stem(video_path: str) -> str:
    """Same stem rule the picker uses: strip extension, collapse remux/h264 copy suffix."""
    parsed = _parse_video_file(video_path)
    if parsed:
        return parsed[0]
    return os.path.splitext(os.path.basename(video_path))[0]


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # Scratch, not output: the results are the `<video>.pred.csv` files
    # written beside each video, so the engine's log and its raw per-frame
    # `result_detection.json` go somewhere disposable. Removed at the end of a
    # clean run; kept, with its path printed, when something failed. The CLI
    # passes its own `work_dir` and manages the same lifetime itself.
    scratch_dir = None
    if args.cfg_options is None or "work_dir" not in args.cfg_options:
        scratch_dir = tempfile.mkdtemp(prefix="trace-predict-")
        cfg.work_dir = scratch_dir

    set_seed(args.seed)
    cfg = update_workdir(cfg)
    create_folder(cfg.work_dir)

    logger = setup_logger("Infer", save_dir=cfg.work_dir)
    logger.debug(f"Using torch version: {torch.__version__}, "
                 f"CUDA version: {torch.version.cuda}")

    # Discover and probe videos
    input_path = os.path.abspath(args.input)
    video_paths = discover_videos(input_path)
    if args.include_stems:
        wanted = set(args.include_stems)
        before = len(video_paths)
        video_paths = [p for p in video_paths if _video_stem(p) in wanted]
        logger.info(f"Filtered {before} → {len(video_paths)} video(s) by stem allowlist")
        if not video_paths:
            raise RuntimeError(
                f"--include-stems matched no videos in {input_path}. "
                f"Requested stems: {sorted(wanted)}"
            )
    logger.info(f"Found {len(video_paths)} video(s)")

    # Determine data_path (directory containing videos)
    if os.path.isfile(input_path):
        data_path = os.path.dirname(input_path)
    else:
        data_path = input_path

    # Generate virtual-clip pseudo-annotations backed by decode proxies.
    from vtrace.proxy_geometry import config_geometry

    clip_frames = int(getattr(cfg.dataset.test, "window_size", 768))
    pseudo_anno = generate_pseudo_annotations(
        video_paths,
        logger,
        cache_dir=cfg.work_dir,
        clip_frames=clip_frames,
        proxy_geometry=config_geometry(cfg, splits=("test",)),
        proxy_workers=_proxy_workers_from_cfg(cfg),
    )

    # Write to a temp file
    tmp_anno_fd, tmp_anno_path = tempfile.mkstemp(suffix=".json", prefix="trace_infer_")
    try:
        with os.fdopen(tmp_anno_fd, "w") as f:
            json.dump(pseudo_anno, f)

        # Override dataset config for inference
        cfg.dataset.test.ann_file = tmp_anno_path
        cfg.dataset.test.data_path = data_path
        cfg.dataset.test.class_map = args.class_map
        cfg.dataset.test.test_mode = True

        # Also override evaluation ground truth if present
        if hasattr(cfg, "evaluation"):
            cfg.evaluation.ground_truth_filename = tmp_anno_path

        # Build dataset
        test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))

        # Auto-detect num_classes from class_map
        num_classes = len(test_dataset.class_map)
        if num_classes_cfg(cfg).num_classes != num_classes:
            logger.info(f"Auto-detected num_classes={num_classes} from class_map "
                        f"(config had {num_classes_cfg(cfg).num_classes}), overriding.")
            num_classes_cfg(cfg).num_classes = num_classes

        # Build model
        model = build_detector(cfg.model)
        model = model.cuda()

        # Load checkpoint
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        logger.debug(f"Checkpoint is epoch {checkpoint['epoch']}.")

        use_ema = getattr(cfg.solver, "ema", False)
        state_key = "state_dict_ema" if use_ema else "state_dict"
        state_dict = checkpoint[state_key]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # backbone.mean/std are constant buffers recreated at init — safe to ignore
        missing = [k for k in missing if k not in ("backbone.mean", "backbone.std")]
        if missing:
            logger.warning(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
        if use_ema:
            logger.debug("Using Model EMA weights.")

        # Auto-tune dataloader parameters
        if args.auto_tune:
            from vtrace.utils import auto_tune_inference
            auto_tune_inference(model, test_dataset, cfg, logger)

        # Build dataloader (after auto-tune so it uses tuned params)
        test_loader = build_dataloader(
            test_dataset,
            shuffle=False,
            drop_last=False,
            **cfg.solver.test,
        )

        # Ensure results are saved to disk
        cfg.post_processing.save_dict = True

        # AMP
        use_amp = getattr(cfg.solver, "amp", False)
        if use_amp:
            logger.debug("Using Automatic Mixed Precision...")

        # Run inference (skip evaluation — no ground truth)
        logger.info("Inference starts ...")
        eval_one_epoch(
            test_loader,
            model,
            cfg,
            logger,
            model_ema=None,
            use_amp=use_amp,
            not_eval=True,  # Always skip evaluation for inference
            profile=args.profile,
            progress_label="predicting",
        )
        logger.info("Inference complete.")
        _shutdown_dataloader_workers(test_loader, logger)

        # Read the result file written by eval_one_epoch
        result_path = os.path.join(cfg.work_dir, "result_detection.json")
        if os.path.isfile(result_path):
            with open(result_path, "r") as f:
                result_data = json.load(f)

            raw_predictions = aggregate_clip_predictions(
                result_data.get("results", {}),
                pseudo_anno["database"],
            )
            predictions = filter_predictions(raw_predictions, args.threshold)

            # One file per video, named after it. `--output` moves the folder;
            # the names stay tied to the videos either way.
            tuned = {}
            if args.tune_threshold:
                for vpath in video_paths:
                    stem = _video_stem(vpath)
                    truth = _truth_bouts(vpath)
                    if not truth:
                        continue
                    try:
                        _frames, duration, _fps, _pts = probe_video(vpath)
                    except Exception:
                        continue
                    # Sweep the unfiltered detections: `predictions` is already
                    # cut at --threshold, which would put a floor under the sweep.
                    tuned[stem] = tune_thresholds(
                        raw_predictions.get(stem, []), truth, duration)
                for stem, cutoffs in tuned.items():
                    if not cutoffs:
                        continue
                    logger.info(f"Tuned thresholds for {stem} "
                                f"(frame-level F1 against its own annotation):")
                    for label, (cutoff, precision, recall, f1) in sorted(cutoffs.items()):
                        logger.info(f"    {label:<16} threshold {cutoff:.2f}  "
                                    f"P {precision:.2f}  R {recall:.2f}  F1 {f1:.2f}")

            written = write_prediction_files(
                video_paths,
                predictions,
                test_dataset.class_map,
                args.threshold,
                output_dir=args.output,
                logger=logger,
                tuned=tuned,
            )
            if args.frame_scores:
                # From the unfiltered detections: the file is for curves over the
                # whole score range, which a cutoff would truncate.
                write_frame_scores(
                    video_paths, raw_predictions, test_dataset.class_map,
                    output_dir=args.output, logger=logger,
                )

            # Summarise what was written: bouts, since that is what the files
            # contain — the raw per-frame count means nothing to a reviewer.
            for video_name, path in written.items():
                bouts = _merge_detections(predictions.get(video_name, [])) \
                    if predictions.get(video_name) else []
                logger.debug(f"  {os.path.basename(path)}: {len(bouts)} bouts")
        else:
            logger.warning("No result file found. Check if post_processing.save_dict is enabled.")

        if scratch_dir:
            # Only reached when nothing above raised.
            shutil.rmtree(scratch_dir, ignore_errors=True)

    except BaseException:
        if scratch_dir and os.path.isdir(scratch_dir):
            print(f"Run scratch kept for inspection: {scratch_dir}", file=sys.stderr)
        raise
    finally:
        # Clean up temp file
        if os.path.exists(tmp_anno_path):
            os.unlink(tmp_anno_path)


if __name__ == "__main__":
    main()
