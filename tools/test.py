import os
import argparse
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trace_tad.config import Config, DictAction
from trace_tad.models import build_detector
from trace_tad.datasets import build_dataset, build_dataloader
from trace_tad.cores import eval_one_epoch
from trace_tad.utils import update_workdir, set_seed, create_folder, setup_logger


def _resolve_eval_subset(ann_file, preferred="test", fallback="validation"):
    """Pick the subset to evaluate for a standalone test/eval run.

    Prefers the held-out ``test`` split (unbiased final reporting) when the
    annotation actually contains test entries; otherwise falls back to
    ``validation`` so legacy 2-way datasets keep working unchanged.
    """
    try:
        import json
        with open(ann_file, "r") as f:
            database = json.load(f).get("database", {})
    except Exception:
        return fallback
    subsets = {v.get("subset") for v in database.values()}
    return preferred if preferred in subsets else fallback


def _load_recommended_global_threshold(*dirs):
    """Return ``(global_threshold, path)`` from the first
    ``recommended_thresholds.json`` found among ``dirs`` (val-tuned at training
    time), or ``(None, None)``."""
    import json
    for d in dirs:
        if not d:
            continue
        path = os.path.join(d, "recommended_thresholds.json")
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    spec = json.load(f)
                g = spec.get("global")
                if g is not None:
                    return float(g), path
            except Exception:
                pass
    return None, None


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="none", help="the checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--not_eval", action="store_true", help="whether to not to eval, only do inference")
    parser.add_argument("--profile", action="store_true", help="enable inference profiling (CPU + GPU timing breakdown)")
    parser.add_argument("--auto-tune", action="store_true", help="auto-tune dataloader params based on system resources")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


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


def _clip_frames_from_cfg(cfg, fallback=768):
    try:
        return int(cfg.dataset.test.window_size)
    except Exception:
        return fallback


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # propagate top-level path overrides into nested dataset/evaluation configs
    if hasattr(cfg, "annotation_path"):
        for split in ("train", "val", "test"):
            if hasattr(cfg.dataset, split):
                cfg.dataset[split].ann_file = cfg.annotation_path
        if hasattr(cfg, "evaluation"):
            cfg.evaluation.ground_truth_filename = cfg.annotation_path
    if hasattr(cfg, "class_map"):
        for split in ("train", "val", "test"):
            if hasattr(cfg.dataset, split):
                cfg.dataset[split].class_map = cfg.class_map
    if hasattr(cfg, "data_path"):
        for split in ("train", "val", "test"):
            if hasattr(cfg.dataset, split):
                cfg.dataset[split].data_path = cfg.data_path

    # set random seed, create work_dir
    set_seed(args.seed)
    cfg = update_workdir(cfg)
    create_folder(cfg.work_dir)

    # setup logger
    logger = setup_logger("Test", save_dir=cfg.work_dir)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: {args.config}")

    # Evaluation uses cached clips too. This keeps direct `trace eval` on older
    # virtual datasets from repeatedly decoding long raw source videos.
    if getattr(cfg.dataset.test, "ann_file", None):
        from trace_tad.data_prep import materialize_dataset_cached_videos

        cached_annotation = materialize_dataset_cached_videos(
            cfg.dataset.test.ann_file,
            cfg.work_dir,
            clip_frames=_clip_frames_from_cfg(cfg),
            cache_resolution=_clip_cache_resolution(cfg),
            cache_workers=_cache_workers_from_cfg(cfg),
            logger=logger,
        )
        cfg.dataset.test.ann_file = cached_annotation
        if hasattr(cfg, "evaluation"):
            cfg.evaluation.ground_truth_filename = cached_annotation

    # Evaluate the held-out 'test' split when the dataset has one (3-way split);
    # fall back to 'validation' for legacy 2-way datasets. Both the loader's
    # subset and the evaluator's subset must agree.
    eval_subset = _resolve_eval_subset(getattr(cfg.dataset.test, "ann_file", None))
    cfg.dataset.test.subset_name = eval_subset
    if hasattr(cfg, "evaluation"):
        cfg.evaluation.subset = eval_subset
    logger.info(f"Evaluating on '{eval_subset}' subset.")

    # build dataset
    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))

    # Auto-detect num_classes from dataset
    num_classes = len(test_dataset.class_map)
    if cfg.model.rpn_head.num_classes != num_classes:
        logger.info(f"Auto-detected num_classes={num_classes} from dataset "
                    f"(config had {cfg.model.rpn_head.num_classes}), overriding.")
        cfg.model.rpn_head.num_classes = num_classes

    # build model
    model = build_detector(cfg.model)
    model = model.cuda()

    checkpoint_path = None
    if cfg.inference.load_from_raw_predictions:
        logger.info(f"Loading from raw predictions: {cfg.inference.fuse_list}")
    else:
        if args.checkpoint != "none":
            checkpoint_path = args.checkpoint
        elif "test_epoch" in cfg.inference.keys():
            checkpoint_path = os.path.join(cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth")
        else:
            checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")
        logger.info("Loading checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        logger.info("Checkpoint is epoch {}.".format(checkpoint["epoch"]))

        # handle checkpoints saved with DDP (module. prefix)
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
            logger.info("Using Model EMA...")

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

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")

    # On the held-out 'test' subset, report precision/recall/F1 at the threshold
    # tuned on validation (the one `trace predict` deploys), not one re-optimized
    # on test — otherwise the held-out P/R/F1 would be optimistically biased. mAP
    # is threshold-free and unaffected. Only when the user didn't pin a threshold.
    if (hasattr(cfg, "evaluation")
            and cfg.evaluation.get("subset") == "test"
            and cfg.evaluation.get("score_threshold") is None):
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path)) if checkpoint_path else None
        parent_dir = os.path.dirname(ckpt_dir) if ckpt_dir else None
        thr, src = _load_recommended_global_threshold(cfg.work_dir, ckpt_dir, parent_dir)
        if thr is not None:
            cfg.evaluation.score_threshold = thr
            logger.info(f"Reporting test precision/recall/F1 at the val-tuned "
                        f"threshold {thr:.2f} (from {src}).")
        else:
            logger.info("No val-tuned threshold found; test precision/recall/F1 "
                        "uses the F1-optimal threshold on test.")

    # test the detector
    logger.info("Testing Starts...\n")
    eval_one_epoch(
        test_loader,
        model,
        cfg,
        logger,
        model_ema=None,
        use_amp=use_amp,
        not_eval=args.not_eval,
        profile=args.profile,
    )
    logger.info("Testing Over...\n")


if __name__ == "__main__":
    main()
