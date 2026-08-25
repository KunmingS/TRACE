import os
import argparse
import sys
import torch
from torch.amp import GradScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vtrace.config import Config, DictAction, num_classes_cfg
from vtrace.model_artifacts import RESOLVED_CONFIG_NAME
from vtrace.models import build_detector
from vtrace.datasets import build_dataset, build_dataloader
from vtrace.cores import train_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from vtrace.utils import (
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
    parser.add_argument("--init_weights", type=str, default=None, help="load model+ema WEIGHTS ONLY from a checkpoint (no optimizer/scheduler/epoch); training starts at epoch 0 with a FRESH schedule. For continued training as a new LR cycle.")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
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

    # Auto-detect num_classes from dataset (DFC-only model: lives at model.num_classes)
    num_classes = len(train_dataset.class_map)
    nc_cfg = num_classes_cfg(cfg)
    if nc_cfg.get("num_classes") != num_classes:
        logger.info(f"Auto-detected num_classes={num_classes} from dataset "
                    f"(config had {nc_cfg.get('num_classes')}), overriding.")
        nc_cfg.num_classes = num_classes

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
        if unexpected:
            # Benign direction: extra entries in the checkpoint are ignored. Pre-DFC-only
            # checkpoints carry the removed localization head (loc_head.* / rpn_head.*);
            # those land here and are simply dropped.
            logger.warning(f"Unexpected keys in checkpoint (ignored): {unexpected}")
        # MISSING is the dangerous direction and is fatal: it means a module in the model
        # got no weights (e.g. a rename left a submodule unmatched), which silently yields
        # a randomly initialised module and plausible-but-wrong numbers.
        if missing:
            raise RuntimeError(
                f"Checkpoint is missing weights for: {missing}\nThese modules would be "
                f"randomly initialised — the checkpoint does not match the model."
            )

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
        # init_weights: load model+ema WEIGHTS ONLY, fresh optimizer/scheduler/epoch
        if args.init_weights is not None:
            logger.info("Init weights (fresh schedule) from: {}".format(args.init_weights))
            ckpt = torch.load(args.init_weights, map_location="cuda")
            sd = ckpt["state_dict"]
            if any(k.startswith("module.") for k in sd.keys()):
                sd = {k.removeprefix("module."): v for k, v in sd.items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            missing = [k for k in missing if k not in ("backbone.mean", "backbone.std")]
            if unexpected:
                logger.warning(f"init_weights unexpected keys (ignored): {unexpected}")
            # See the resume branch: missing weights are fatal, extra ones are not.
            if missing:
                raise RuntimeError(
                    f"init_weights checkpoint is missing weights for: {missing}\n"
                    f"These modules would be randomly initialised."
                )
            if model_ema is not None and "state_dict_ema" in ckpt:
                es = ckpt["state_dict_ema"]
                if any(k.startswith("module.") for k in es.keys()):
                    es = {k.removeprefix("module."): v for k, v in es.items()}
                model_ema.module.load_state_dict(es)
            logger.info(f"init_weights loaded (from epoch {ckpt.get('epoch')}); training fresh from epoch 0.")
            del ckpt
            torch.cuda.empty_cache()

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

        # The config the run actually used: bases merged, every --cfg-options
        # override applied, num_classes as detected from the data. Inference
        # prefers this over config.txt, which is only a path and would follow
        # later edits of the source file.
        cfg.dump(os.path.join(model_dir, RESOLVED_CONFIG_NAME))

        # Kept for tools that still expect it, and as a record of where this run
        # started from.
        with open(os.path.join(model_dir, "config.txt"), "w") as f:
            f.write(args.config + "\n")

        logger.info(f"Model folder saved to: {model_dir}")
        logger.info(f"  best.pth + classmap.txt + {RESOLVED_CONFIG_NAME} are ready")
    else:
        logger.warning("No best.pth found — evaluation may not have run. "
                       "Model folder not created.")


    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
