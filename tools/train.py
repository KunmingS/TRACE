import os
import argparse
import sys
import torch
from torch.amp import GradScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vtrace.config import Config, DictAction, num_classes_cfg
from vtrace.console import ACCENT, say
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
from vtrace.utils.logger import BRIEF
from vtrace.utils.progress import EpochBar, LossPlot, plot_width
from vtrace.verbosity import verbose


# Warnings that fire on every run, say nothing about this run, and are four
# lines of stack trace each. They stay one `TRACE_VERBOSE=1` away.
_EXPECTED_WARNINGS = (
    "torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly",
    "Flash Attention defaults to a non-deterministic algorithm",
    "upsample_linear1d_backward_out_cuda does not have a deterministic implementation",
    "Detected call of `lr_scheduler.step()` before `optimizer.step()`",
)


def _quiet_expected_warnings():
    import re
    import warnings

    for message in _EXPECTED_WARNINGS:
        warnings.filterwarnings("ignore", message=re.escape(message))


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


def _latest_epoch_checkpoint(work_dir):
    """The highest-numbered `checkpoint/epoch_N.pth`, or None if there is none.

    By epoch number rather than mtime: a resumed run rewrites earlier epochs
    later, and the newest file is then not the furthest-trained one.
    """
    import re

    checkpoint_dir = os.path.join(work_dir, "checkpoint")
    best = (None, None)
    try:
        names = os.listdir(checkpoint_dir)
    except OSError:
        return None
    for name in names:
        match = re.fullmatch(r"epoch_(\d+)\.pth", name)
        if match:
            epoch = int(match.group(1))
            if best[0] is None or epoch > best[0]:
                best = (epoch, os.path.join(checkpoint_dir, name))
    return best[1]


def _print_header(cfg, train_dataset, test_dataset, train_loader,
                  max_epoch, accumulation_steps, use_amp, use_ema):
    """The four facts worth knowing before a run that takes hours.

    Everything else the run knows about itself — the sampler's window table,
    the resolved config, which weights loaded — is in the log file named here.
    """
    setup = [f"[{ACCENT}]{max_epoch}[/] epochs [dim]x[/] "
             f"[{ACCENT}]{len(train_loader)}[/] steps"]
    if accumulation_steps > 1:
        setup.append(f"grad-accum [{ACCENT}]x{accumulation_steps}[/]")
    if use_amp:
        setup.append("mixed precision")
    if use_ema:
        setup.append("EMA")
    try:
        device = torch.cuda.get_device_name(0)
    except Exception:
        device = None

    windows = f"[{ACCENT}]{len(train_dataset)}[/] training"
    if test_dataset is not None:
        windows += f", [{ACCENT}]{len(test_dataset)}[/] validation"

    behaviors = ", ".join(f"[{ACCENT}]{name}[/]"
                          for name in train_dataset.class_map)
    say()
    say("[bold]Training the model[/]")
    say(f"  [dim]behaviors[/]  {behaviors}")
    say(f"  [dim]windows[/]    {windows}")
    joined = " [dim]\u00b7[/] ".join(setup)
    say(f"  [dim]setup[/]      {joined}"
        + (f" [dim]on[/] [dim]{device}[/]" if device else ""))
    say(f"  [dim]log[/]        [dim]{os.path.join(cfg.work_dir, 'log.json')}[/]")
    say()


def main():
    args = parse_args()
    if not verbose():
        _quiet_expected_warnings()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Propagate top-level path overrides into the nested dataset/evaluation
    # configs. Training data and evaluation data are separate corpora, each with
    # its own dataset.json, so the train split and the val/test splits are
    # pointed at different files. `eval_annotation_path` unset means no
    # evaluation data was given, and there is nothing to validate against.
    eval_annotation_path = getattr(cfg, "eval_annotation_path", None)
    eval_data_path = getattr(cfg, "eval_data_path", None) or getattr(cfg, "data_path", None)
    EVAL_SPLITS = ("val", "test")

    if hasattr(cfg, "annotation_path"):
        if hasattr(cfg.dataset, "train"):
            cfg.dataset.train.ann_file = cfg.annotation_path
    if eval_annotation_path:
        for split in EVAL_SPLITS:
            if hasattr(cfg.dataset, split):
                cfg.dataset[split].ann_file = eval_annotation_path
        if hasattr(cfg, "evaluation"):
            cfg.evaluation.ground_truth_filename = eval_annotation_path
    if hasattr(cfg, "class_map"):
        for split in ("train", *EVAL_SPLITS):
            if hasattr(cfg.dataset, split):
                cfg.dataset[split].class_map = cfg.class_map
    if hasattr(cfg, "data_path") and hasattr(cfg.dataset, "train"):
        cfg.dataset.train.data_path = cfg.data_path
    if eval_data_path:
        for split in EVAL_SPLITS:
            if hasattr(cfg.dataset, split):
                cfg.dataset[split].data_path = eval_data_path

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    cfg = update_workdir(cfg)
    create_folder(cfg.work_dir)
    save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, brief=True)
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

    if eval_annotation_path:
        test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
        test_loader = build_dataloader(
            test_dataset,
            shuffle=False,
            drop_last=False,
            **cfg.solver.test,
        )
    else:
        # Training on its own: every epoch is checkpointed and none is called
        # best, because nothing measured them. Said once, here, rather than
        # left for the reader to infer from a missing file at the end.
        test_dataset = test_loader = None
        logger.info(
            "No evaluation data — training only. Every epoch is checkpointed and "
            "no best.pth is written. To pick a best epoch, give evaluation videos: "
            "`vtrace train ... --eval-pairs VIDEO=CSV`, or `train ... then eval "
            "--pairs VIDEO=CSV` at the prompt.",
            extra=BRIEF,
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
        logger.info("Resume training from: {}".format(args.resume), extra=BRIEF)
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
            logger.info(f"init_weights loaded (from epoch {ckpt.get('epoch')}); training fresh from epoch 0.",
                        extra=BRIEF)
            del ckpt
            torch.cuda.empty_cache()

    # train the detector

    logger.info("Training Starts...\n")
    _print_header(cfg, train_dataset, test_dataset, train_loader,
                  max_epoch, accumulation_steps, use_amp, use_ema)
    val_start_epoch = cfg.workflow.get("val_start_epoch", 0)
    best_metric = -1.0
    # One plot for the run, so the curve carries across epochs instead of
    # restarting — the question it answers is whether the loss is still falling.
    plot = LossPlot(plot_width())
    for epoch in range(resume_epoch + 1, max_epoch):
        # train for one epoch. The bar is the terminal's whole view of the
        # epoch; the periodic loss/lr/memory lines still go to the log file.
        bar = EpochBar(epoch, max_epoch, len(train_loader), plot=plot)
        try:
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
                progress=bar.update,
            )
        finally:
            bar.close()

        # save checkpoint
        if (epoch == max_epoch - 1) or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0):
            save_checkpoint(model, model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)

        # eval for one epoch
        if test_loader is not None and epoch >= val_start_epoch:
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
                if primary_metric is not None:
                    improved = primary_metric > best_metric
                    if improved:
                        logger.info(f"New best metric: {primary_metric:.4f}, saving best checkpoint...")
                        save_best_checkpoint(model, model_ema, epoch, work_dir=cfg.work_dir)
                    # One line for what the epoch scored, next to the bar that
                    # produced it. The full per-class table is in the log file.
                    note = (f"[{ACCENT}]best so far[/]" if improved
                            else f"[dim]best {best_metric:.3f}[/]")
                    say(f"  [dim]validation[/]  mAP [bold]{primary_metric:.3f}[/]  {note}")
                    if improved:
                        best_metric = primary_metric
    # Make the work_dir itself a self-contained model folder. A run with no
    # evaluation data still produces a usable model — it just has no epoch that
    # was measured to be better than the others, so the weights it publishes are
    # the last ones, under a name that says so.
    import shutil

    model_dir = os.path.abspath(cfg.work_dir)
    best_pth = os.path.join(cfg.work_dir, "checkpoint", "best.pth")
    if os.path.isfile(best_pth):
        published, label = best_pth, "best.pth"
    else:
        published, label = _latest_epoch_checkpoint(cfg.work_dir), "last.pth"

    if published is None:
        logger.warning("No checkpoint was written — model folder not created.")
    else:
        destination = os.path.join(model_dir, label)
        if os.path.abspath(published) != os.path.abspath(destination):
            shutil.copy2(published, destination)

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
        logger.info(f"  {label} + classmap.txt + {RESOLVED_CONFIG_NAME} are ready")
        if label == "last.pth":
            logger.info("  (last.pth, not best.pth: no evaluation data, so no epoch was scored)")


    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
