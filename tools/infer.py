"""Standalone inference script for TRACE.

Runs temporal action detection on video files without requiring annotation JSONs.
Automatically probes videos for frame count and duration, generates a temporary
annotation file, and outputs predictions as JSON.

Usage:
    python tools/infer.py configs/tridet/tridet_small.py \
        --checkpoint exps/tridet_small/checkpoint_best.pth \
        --input /path/to/videos \
        --class-map data/CALMS21/category_idx.txt

    python tools/infer.py configs/tridet/tridet_small.py \
        --checkpoint exps/tridet_small/checkpoint_best.pth \
        --input /path/to/single_video.mp4 \
        --class-map data/CALMS21/category_idx.txt \
        --output predictions.json
"""
import argparse
import json
import os
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
from trace_tad.utils import set_seed, update_workdir, create_folder, setup_logger


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def probe_video(filepath):
    """Get frame count and duration from a video file using decord."""
    from decord import VideoReader

    vr = VideoReader(filepath)
    num_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = num_frames / fps if fps > 0 else 0.0
    return num_frames, duration, fps


def discover_videos(input_path):
    """Discover video files from a path (file or directory)."""
    videos = []
    if os.path.isfile(input_path):
        videos.append(input_path)
    elif os.path.isdir(input_path):
        for fname in sorted(os.listdir(input_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                videos.append(os.path.join(input_path, fname))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not videos:
        raise FileNotFoundError(f"No video files found in {input_path}")
    return videos


def generate_pseudo_annotations(video_paths, logger):
    """Probe videos and create a temporary annotation JSON for inference."""
    database = {}
    for vpath in video_paths:
        video_name = os.path.splitext(os.path.basename(vpath))[0]
        logger.info(f"Probing video: {video_name}")
        try:
            num_frames, duration, fps = probe_video(vpath)
        except Exception as e:
            logger.warning(f"Skipping {video_name}: {e}")
            continue

        database[video_name] = {
            "subset": "validation",
            "frame": num_frames,
            "duration": duration,
            "annotations": [],
        }
        logger.info(f"  {num_frames} frames, {duration:.2f}s, {fps:.1f} fps")

    if not database:
        raise RuntimeError("No videos could be probed successfully")

    return {"database": database}


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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--id", type=int, default=0, help="Repeat experiment ID")
    parser.add_argument("--profile", action="store_true",
                        help="Enable inference profiling (CPU + GPU timing)")
    parser.add_argument("--auto-tune", action="store_true",
                        help="Auto-tune dataloader params based on system resources")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction,
                        help="Override config settings (key=value pairs)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    set_seed(args.seed)
    cfg = update_workdir(cfg, args.id, 1)
    create_folder(cfg.work_dir)

    logger = setup_logger("Infer", save_dir=cfg.work_dir)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")

    # Discover and probe videos
    input_path = os.path.abspath(args.input)
    video_paths = discover_videos(input_path)
    logger.info(f"Found {len(video_paths)} video(s)")

    # Determine data_path (directory containing videos)
    if os.path.isfile(input_path):
        data_path = os.path.dirname(input_path)
    else:
        data_path = input_path

    # Generate pseudo-annotations
    pseudo_anno = generate_pseudo_annotations(video_paths, logger)

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
            predictions = result_data.get("results", {})
            output = {
                "num_videos": len(predictions),
                "class_map": test_dataset.class_map,
                "predictions": predictions,
            }

            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Predictions saved to: {output_path}")

            # Also save as CSV (same format as annotation CSVs)
            csv_path = os.path.splitext(output_path)[0] + ".csv"
            with open(csv_path, "w") as f:
                f.write("labelId,timestamp,endTimestamp\n")
                for dets in predictions.values():
                    for det in sorted(dets, key=lambda d: d["segment"][0]):
                        f.write(f"{det['label']},{det['segment'][0]:.3f},{det['segment'][1]:.3f}\n")
            logger.info(f"CSV saved to: {csv_path}")

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
