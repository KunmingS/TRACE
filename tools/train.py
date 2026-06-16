import os
import argparse
import json
import sys
import torch
from torch.amp import GradScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trace_tad.config import Config, DictAction
from trace_tad.models import build_detector
from trace_tad.datasets import build_dataset, build_dataloader
from trace_tad.cores import train_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from trace_tad.utils import (
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


def _save_recommended_thresholds(work_dir, logger=None):
    """Flatten this epoch's recommended thresholds from metrics.json and pin
    them next to best.pth as recommended_thresholds.json.

    eval_one_epoch writes metrics.json for the just-finished validation eval, so
    calling this right after a new-best checkpoint save captures the thresholds
    tuned at the best epoch. `trace predict` auto-loads this file.
    """
    metrics_path = os.path.join(work_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        return
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except Exception:
        return
    rec = metrics.get("recommended_thresholds")
    if not rec:
        return
    flat = {
        "global": rec.get("global", {}).get("threshold", 0.0),
        "per_class": {
            label: info.get("threshold")
            for label, info in (rec.get("per_class") or {}).items()
        },
    }
    out_path = os.path.join(work_dir, "recommended_thresholds.json")
    with open(out_path, "w") as f:
        json.dump(flat, f, indent=2)
    if logger:
        logger.info(f"Saved recommended thresholds to {out_path}: {flat}")


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

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    cfg = update_workdir(cfg)
    create_folder(cfg.work_dir)
    save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: {args.config}")

    # build dataset
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    train_loader = build_dataloader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )

    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    # Auto-compute samples_per_class for ClassBalancedFocalLoss
    cls_loss_cfg = cfg.model.get("rpn_head", {}).get("loss", {}).get("cls_loss", {})
    if cls_loss_cfg.get("type") == "ClassBalancedFocalLoss" and cls_loss_cfg.get("samples_per_class") is None:
        from trace_tad.models.losses.focal_loss import count_samples_per_class
        ann_file = cfg.dataset.train.ann_file
        class_map = train_dataset.class_map
        subset_name = cfg.dataset.train.get("subset_name", "training")
        samples_per_class = count_samples_per_class(ann_file, class_map, subset_name)
        cfg.model.rpn_head.loss.cls_loss.samples_per_class = samples_per_class
        logger.info(f"Auto-computed samples_per_class: {samples_per_class}")

    # Auto-detect num_classes from dataset
    num_classes = len(train_dataset.class_map)
    if cfg.model.rpn_head.num_classes != num_classes:
        logger.info(f"Auto-detected num_classes={num_classes} from dataset "
                    f"(config had {cfg.model.rpn_head.num_classes}), overriding.")
        cfg.model.rpn_head.num_classes = num_classes

    # build model
    model = build_detector(cfg.model)
    model = model.cuda()

    # torch.compile for speedup on Ampere+ GPUs
    if getattr(cfg.solver, "compile", False):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        logger.info("Using Model EMA...")
        model_ema = ModelEma(model)
    else:
        model_ema = None

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        scaler = GradScaler("cuda")
    else:
        scaler = None

    # gradient accumulation setup
    accumulation_steps = getattr(cfg.solver, "accumulation_steps", 1)
    if accumulation_steps > 1:
        logger.info(f"Using gradient accumulation: {accumulation_steps} steps (effective batch size x{accumulation_steps})")

    # build optimizer and scheduler
    # With gradient accumulation, the scheduler steps fewer times per epoch
    optimizer = build_optimizer(cfg.optimizer, model, logger)
    steps_per_epoch = -(-len(train_loader) // accumulation_steps)  # ceil division
    scheduler, max_epoch = build_scheduler(cfg.scheduler, optimizer, steps_per_epoch)

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    # resume: reset epoch, load checkpoint
    if args.resume is not None:
        logger.info("Resume training from: {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location="cuda")
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))

        # handle checkpoints saved with DDP (module. prefix)
        state_dict = checkpoint["state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if k not in ("backbone.mean", "backbone.std")]
        if missing:
            logger.warning(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected}")

        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if model_ema is not None and "state_dict_ema" in checkpoint:
            ema_state = checkpoint["state_dict_ema"]
            if any(k.startswith("module.") for k in ema_state.keys()):
                ema_state = {k.removeprefix("module."): v for k, v in ema_state.items()}
            model_ema.module.load_state_dict(ema_state)

        del checkpoint
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    # train the detector

    logger.info("Training Starts...\n")
    val_start_epoch = cfg.workflow.get("val_start_epoch", 0)
    best_metric = -1.0
    for epoch in range(resume_epoch + 1, max_epoch):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            logger,
            model_ema=model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=cfg.workflow.logging_interval,
            scaler=scaler,
            accumulation_steps=accumulation_steps,
        )

        # save checkpoint
        if (epoch == max_epoch - 1) or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0):
            save_checkpoint(model, model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)

        # eval for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_eval_interval > 0) and ((epoch + 1) % cfg.workflow.val_eval_interval == 0):
                primary_metric = eval_one_epoch(
                    test_loader,
                    model,
                    cfg,
                    logger,
                    model_ema=model_ema,
                    use_amp=use_amp,
                    not_eval=args.not_eval,
                )

                # save best model
                if primary_metric is not None and primary_metric > best_metric:
                    best_metric = primary_metric
                    logger.info(f"New best metric: {best_metric:.4f}, saving best checkpoint...")
                    save_best_checkpoint(model, model_ema, epoch, work_dir=cfg.work_dir)
                    # Pin THIS epoch's val-tuned recommended thresholds next to
                    # best.pth so `trace predict` can apply them automatically.
                    _save_recommended_thresholds(cfg.work_dir, logger)
    # Make the work_dir itself a self-contained model folder.
    best_pth = os.path.join(cfg.work_dir, "checkpoint", "best.pth")
    if os.path.isfile(best_pth):
        model_dir = os.path.abspath(cfg.work_dir)
        root_best = os.path.join(model_dir, "best.pth")
        if os.path.abspath(best_pth) != os.path.abspath(root_best):
            import shutil
            shutil.copy2(best_pth, root_best)

        # Generate classmap from the training dataset's class_map list.
        with open(os.path.join(model_dir, "classmap.txt"), "w") as f:
            for name in train_dataset.class_map:
                f.write(name + "\n")

        # Save config path so model folder is self-contained.
        with open(os.path.join(model_dir, "config.txt"), "w") as f:
            f.write(args.config + "\n")

        logger.info(f"Model folder saved to: {model_dir}")
        logger.info("  best.pth + dataset.json + classmap.txt + config.txt are ready")
    else:
        logger.warning("No best.pth found — evaluation may not have run. "
                       "Model folder not created.")

    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
