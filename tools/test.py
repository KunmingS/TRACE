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


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="none", help="the checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--not_eval", action="store_true", help="whether to not to eval, only do inference")
    parser.add_argument("--profile", action="store_true", help="enable inference profiling (CPU + GPU timing breakdown)")
    parser.add_argument("--auto-tune", action="store_true", help="auto-tune dataloader params based on system resources")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


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
    cfg = update_workdir(cfg, args.id, 1)
    create_folder(cfg.work_dir)

    # setup logger
    logger = setup_logger("Test", save_dir=cfg.work_dir)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: {args.config}")

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
