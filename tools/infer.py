"""Standalone inference script for TRACE.

Runs temporal action detection on video files without requiring annotation JSONs.
Automatically probes videos for frame count, duration, and a per-frame PTS
table (cached as ``<video>.pts.npy`` next to the source — see
``docs/pts-based-frame-mapping.md``). Before running the model, source videos
are split into small cached clips under the prediction work directory so
inference workers do not keep decoding long raw files.

Usage:
    python tools/infer.py configs/small.py \
        --checkpoint exps/small/checkpoint_best.pth \
        --input /path/to/videos \
        --class-map data/CALMS21/category_idx.txt

    python tools/infer.py configs/small.py \
        --checkpoint exps/small/checkpoint_best.pth \
        --input /path/to/single_video.mp4 \
        --class-map data/CALMS21/category_idx.txt \
        --output predictions.json
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trace_tad.config import Config, DictAction
from trace_tad.models import build_detector
from trace_tad.datasets import build_dataset, build_dataloader
from trace_tad.cores import eval_one_epoch
from trace_tad.model_artifacts import create_predict_dir_for_input
from trace_tad.utils import set_seed, update_workdir, create_folder, setup_logger
from trace_tad.video_annotation import filter_predictions, render_annotated_videos


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
      the convention used by ``trace_tad.data_prep`` so train, eval, and
      inference all agree on clip length.
    - ``fps`` — average fps from decord (display-only; no time ↔ frame
      math goes through it once the PTS table is available). Falls back
      to PTS-span estimation if the container reports an invalid value.
    - ``pts_path`` — absolute path of the cached ``<video>.pts.npy`` if
      it was successfully written, else ``None``.

    See ``docs/pts-based-frame-mapping.md`` for the design and why this
    matters for VFR webcam recordings.
    """
    from decord import VideoReader
    from trace_tad.data_prep import _load_or_build_pts, _pts_cache_path

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


def _clip_cache_resolution(cfg, fallback=144):
    try:
        for step in cfg.dataset.test.pipeline:
            if step.get("type") != "VideoInit":
                continue
            resize = step.get("resize", None)
            if isinstance(resize, int):
                return resize
            if isinstance(resize, (list, tuple)) and resize:
                return int(resize[0])
    except Exception:
        pass
    return fallback


def _cache_workers_from_cfg(cfg, fallback=4):
    try:
        return int(cfg.solver.test.num_workers)
    except Exception:
        return fallback


def generate_pseudo_annotations(
    video_paths,
    logger,
    cache_dir=None,
    clip_frames=768,
    cache_resolution=144,
    cache_crf=23,
    cache_workers=None,
):
    """Probe videos and create a temporary annotation JSON for inference.

    Each source video is materialized into cached clips. The dataset decodes
    from those smaller files while preserving ``source_video`` /
    ``source_frame_offset`` / ``source_pts_table`` metadata for PTS-aware
    clip-local second conversion.
    """
    from trace_tad.data_prep import materialize_video_clips

    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="trace_infer_cache_")

    database = {}
    for vpath in video_paths:
        video_name = _video_stem(vpath)
        logger.info(f"Probing video: {video_name}")
        try:
            num_frames, duration, fps, pts_path = probe_video(vpath)
        except Exception as e:
            logger.warning(f"Skipping {video_name}: {e}")
            continue

        logger.info(
            f"  {num_frames} frames, {duration:.2f}s, {fps:.1f} fps"
            + (f", PTS cached" if pts_path else f", PTS cache unavailable (read-only dir?)")
        )
        try:
            clips = materialize_video_clips(
                vpath,
                cache_dir,
                clip_frames=clip_frames,
                cache_resolution=cache_resolution,
                cache_crf=cache_crf,
                cache_workers=cache_workers,
                clip_stem=video_name,
                logger=logger,
            )
        except Exception as e:
            logger.warning(f"Skipping {video_name}: could not cache clips: {e}")
            continue

        cached_count = sum(1 for clip in clips if "cached_video" in clip)
        logger.info(f"  Wrote {cached_count}/{len(clips)} cached inference clip(s)")
        for clip in clips:
            clip_idx = int(clip["clip_idx"])
            clip_key = f"{video_name}_clip_{clip_idx}"
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
            if "cached_video" in clip:
                entry["cached_video"] = clip["cached_video"]
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


def _write_prediction_csv(csv_path, detections):
    with open(csv_path, "w") as f:
        f.write("labelId,timestamp,endTimestamp\n")
        for det in sorted(detections, key=lambda d: d["segment"][0]):
            f.write(f"{det['label']},{det['segment'][0]:.3f},{det['segment'][1]:.3f}\n")


def _adjacent_prediction_csv_path(video_path, video_name):
    video_dir = os.path.dirname(os.path.abspath(video_path))
    canonical = os.path.join(video_dir, f"{video_name}.csv")
    if not os.path.exists(canonical):
        return canonical
    return os.path.join(video_dir, f"{video_name}_predicted.csv")


def write_prediction_csvs(video_paths, predictions, output_dir, logger=None):
    """Write one prediction CSV per source video and copy it next to the video.

    The prediction work directory always gets ``{video_stem}.csv``. Next to
    the source video we use the same name when it is free, so the annotator can
    pair video + CSV directly. If a manual ``{video_stem}.csv`` already exists,
    copy to ``{video_stem}_predicted.csv`` instead; that still appears as a CSV
    variant for the video without overwriting user annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    prediction_csvs = {}
    adjacent_csvs = {}

    for video_path in video_paths:
        video_name = _video_stem(video_path)
        detections = predictions.get(video_name, [])
        preferred_csv_path = os.path.join(output_dir, f"{video_name}.csv")
        adjacent_path = _adjacent_prediction_csv_path(video_path, video_name)
        csv_path = preferred_csv_path
        preferred_dir = os.path.dirname(os.path.abspath(preferred_csv_path))
        adjacent_dir = os.path.dirname(os.path.abspath(adjacent_path))
        if (
            os.path.abspath(preferred_csv_path) != os.path.abspath(adjacent_path)
            and preferred_dir == adjacent_dir
        ):
            csv_path = adjacent_path
        _write_prediction_csv(csv_path, detections)
        prediction_csvs[video_name] = csv_path

        if os.path.abspath(adjacent_path) != os.path.abspath(csv_path):
            if logger and os.path.basename(adjacent_path) != f"{video_name}.csv":
                logger.info(
                    f"Existing annotation CSV found for {video_name}; "
                    f"copying predictions as {os.path.basename(adjacent_path)}"
                )
            shutil.copyfile(csv_path, adjacent_path)
        adjacent_csvs[video_name] = adjacent_path

        if logger:
            logger.info(f"Prediction CSV saved to: {csv_path}")
            logger.info(f"Prediction CSV copied to: {adjacent_path}")

    return prediction_csvs, adjacent_csvs


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
                        help="Output JSON path (default: predictions.json in work_dir)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Minimum score for predictions.json, CSV, and annotated "
                             "videos. If omitted, TRACE auto-applies the per-class "
                             "F1-optimal thresholds recommended during training "
                             "(recommended_thresholds.json next to the checkpoint), "
                             "falling back to 0.0 when none are available.")
    parser.add_argument("--annotated-video", action="store_true",
                        help="Render predictions onto annotated MP4 videos in work_dir")
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
    if args.threshold is not None and (args.threshold < 0.0 or args.threshold > 1.0):
        parser.error("--threshold must be between 0 and 1")
    return args


def _resolve_inference_threshold(args, logger):
    """Resolve the score threshold(s) used to filter predictions.

    An explicit ``--threshold`` always wins (scalar, applied to every class).
    Otherwise TRACE looks for ``recommended_thresholds.json`` (written next to
    the best checkpoint during training, tuned on validation) and applies its
    per-class thresholds, falling back to 0.0 when no such file exists.
    """
    if args.threshold is not None:
        logger.info(f"Using explicit --threshold {args.threshold:.2f} for all classes.")
        return float(args.threshold)

    ckpt = getattr(args, "checkpoint", None)
    candidates = []
    if ckpt and ckpt != "none":
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt))
        candidates = [
            os.path.join(ckpt_dir, "recommended_thresholds.json"),
            os.path.join(os.path.dirname(ckpt_dir), "recommended_thresholds.json"),
        ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    spec = json.load(f)
            except Exception as exc:
                logger.warning(f"Could not read {path}: {exc}; using threshold 0.0.")
                return 0.0
            logger.info(
                f"Auto-applying recommended thresholds from {path}: "
                f"global={spec.get('global')}, per_class={spec.get('per_class')}"
            )
            return spec

    logger.info("No recommended_thresholds.json found; using threshold 0.0 (no score filtering).")
    return 0.0


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
    if args.cfg_options is None or "work_dir" not in args.cfg_options:
        cfg.work_dir = create_predict_dir_for_input(args.input)

    set_seed(args.seed)
    cfg = update_workdir(cfg)
    create_folder(cfg.work_dir)

    logger = setup_logger("Infer", save_dir=cfg.work_dir)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")

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

    # Generate pseudo-annotations backed by cached clips.
    clip_frames = int(getattr(cfg.dataset.test, "window_size", 768))
    cache_resolution = _clip_cache_resolution(cfg)
    pseudo_anno = generate_pseudo_annotations(
        video_paths,
        logger,
        cache_dir=cfg.work_dir,
        clip_frames=clip_frames,
        cache_resolution=cache_resolution,
        cache_workers=_cache_workers_from_cfg(cfg),
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
        if cfg.model.rpn_head.num_classes != num_classes:
            logger.info(f"Auto-detected num_classes={num_classes} from class_map "
                        f"(config had {cfg.model.rpn_head.num_classes}), overriding.")
            cfg.model.rpn_head.num_classes = num_classes

        # Build model
        model = build_detector(cfg.model)
        model = model.cuda()

        # Load checkpoint
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        logger.info(f"Checkpoint is epoch {checkpoint['epoch']}.")

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
            logger.info("Using Model EMA weights.")

        # Auto-tune dataloader parameters
        if args.auto_tune:
            from trace_tad.utils import auto_tune_inference
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
            logger.info("Using Automatic Mixed Precision...")

        # Run inference (skip evaluation — no ground truth)
        logger.info("Inference starts...\n")
        eval_one_epoch(
            test_loader,
            model,
            cfg,
            logger,
            model_ema=None,
            use_amp=use_amp,
            not_eval=True,  # Always skip evaluation for inference
            profile=args.profile,
        )
        logger.info("Inference complete.\n")
        _shutdown_dataloader_workers(test_loader, logger)

        # Read the result file written by eval_one_epoch
        result_path = os.path.join(cfg.work_dir, "result_detection.json")
        if os.path.isfile(result_path):
            with open(result_path, "r") as f:
                result_data = json.load(f)

            # Determine output path
            output_path = args.output
            if output_path is None:
                output_path = os.path.join(cfg.work_dir, "predictions.json")

            # Reformat for user-friendly output
            raw_predictions = aggregate_clip_predictions(
                result_data.get("results", {}),
                pseudo_anno["database"],
            )
            threshold_spec = _resolve_inference_threshold(args, logger)
            predictions = filter_predictions(raw_predictions, threshold_spec)
            output = {
                "num_videos": len(predictions),
                "class_map": test_dataset.class_map,
                "threshold": threshold_spec,
                "predictions": predictions,
            }

            prediction_csvs, adjacent_csvs = write_prediction_csvs(
                video_paths,
                predictions,
                cfg.work_dir,
                logger=logger,
            )
            output["prediction_csvs"] = {
                name: os.path.basename(path)
                for name, path in prediction_csvs.items()
            }
            output["adjacent_prediction_csvs"] = adjacent_csvs

            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Predictions saved to: {output_path}")

            if args.annotated_video:
                annotated_paths = render_annotated_videos(
                    video_paths,
                    predictions,
                    cfg.work_dir,
                    threshold=threshold_spec,
                    logger=logger,
                )
                output["annotated_videos"] = {
                    name: os.path.basename(path)
                    for name, path in annotated_paths.items()
                }
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=2)
                logger.info(f"Predictions updated with annotated video paths: {output_path}")

            # Print summary
            total_detections = sum(len(dets) for dets in predictions.values())
            logger.info(f"Total: {len(predictions)} videos, {total_detections} detections")
            for video_name, dets in predictions.items():
                logger.info(f"  {video_name}: {len(dets)} detections")
                for det in dets[:5]:  # Show first 5
                    logger.info(
                        f"    [{det['segment'][0]:.2f}s - {det['segment'][1]:.2f}s] "
                        f"{det['label']} (score={det['score']:.3f})"
                    )
                if len(dets) > 5:
                    logger.info(f"    ... and {len(dets) - 5} more")
        else:
            logger.warning("No result file found. Check if post_processing.save_dict is enabled.")

    finally:
        # Clean up temp file
        if os.path.exists(tmp_anno_path):
            os.unlink(tmp_anno_path)


if __name__ == "__main__":
    main()
