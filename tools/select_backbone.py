"""Fast, training-free backbone selection for a TRACE dataset.

Given several candidate backbone *configs* (same dataset, different pretrained
backbone), this ranks them by transferability metrics computed on the pure
frozen-backbone features — NO training. Pick the winner, then full-train only
that one (with the adapter live). See ``model-selection-recipe.md (research notes, archived)``.

Each candidate config must be self-contained (dataset + backbone). The dataset
is read from each config's ``dataset.test`` (validation) split. The adapter is
force-disabled (``adapter_index=[]``) so features are head-independent.

Usage:
    python tools/select_backbone.py \
        --configs configs/calms21_vitB_mae_multiscale.py:K710:/path/k710.pth \
                  configs/calms21_vitB_mae_multiscale.py:K400:/path/k400.pth \
                  configs/calms21_vjepa21_multiscale.py:VJEPA21 \
        --methods logme hscore transrate gbc \
        --max-frames 40000 --cache-dir /tmp/sel_feats

A candidate is ``CONFIG[:LABEL[:PRETRAIN_PATH]]``. If PRETRAIN_PATH is given it
overrides ``model.backbone.custom.pretrain``.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtrace.config import Config, num_classes_cfg
from vtrace.datasets import build_dataset, build_dataloader
from vtrace.models import build_detector
from vtrace.selection.extract import extract_features
from vtrace.selection.transferability import METHODS, score_transferability
from vtrace.utils import setup_logger


def parse_candidate(spec):
    """``CONFIG[:LABEL[:PRETRAIN]]`` -> (config, label, pretrain_or_None)."""
    parts = spec.split(":")
    cfg = parts[0]
    label = parts[1] if len(parts) > 1 and parts[1] else os.path.basename(cfg).replace(".py", "")
    pretrain = parts[2] if len(parts) > 2 and parts[2] else None
    return cfg, label, pretrain


def build_one(config_path, pretrain, logger):
    cfg = Config.fromfile(config_path)
    # force head-independent, fully-frozen backbone features
    if hasattr(cfg.model, "backbone"):
        cfg.model.backbone["adapter_index"] = []
    if pretrain is not None:
        cfg.model.backbone.setdefault("custom", {})
        cfg.model.backbone["custom"]["pretrain"] = pretrain

    # use the TEST split: sliding window over original videos, includes ALL
    # windows (good background coverage) and still carries metas
    # (video_name/window_start_frame). Labels are read from the annotation JSON,
    # so we don't depend on the pipeline collecting gt_segments.
    split = cfg.dataset.test
    dataset = build_dataset(split, default_args=dict(logger=logger))
    loader = build_dataloader(dataset, shuffle=False, drop_last=False, **cfg.solver.test)

    num_classes = len(dataset.class_map)
    if num_classes_cfg(cfg).num_classes != num_classes:
        num_classes_cfg(cfg).num_classes = num_classes

    ann_file = split["ann_file"]
    class_map = split["class_map"]
    model = build_detector(cfg.model).cuda().eval()
    return model, loader, num_classes, ann_file, class_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True,
                    help="candidate specs CONFIG[:LABEL[:PRETRAIN]]")
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    choices=list(METHODS))
    ap.add_argument("--max-frames", type=int, default=40000)
    ap.add_argument("--per-clip-cap", type=int, default=256)
    ap.add_argument("--bg-fraction", type=float, default=0.5)
    ap.add_argument("--cache-dir", default=None,
                    help="if set, cache/reuse extracted features as .npz here")
    args = ap.parse_args()

    logger = setup_logger("SelectBackbone", save_dir=None)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    rows = []
    timings = {}
    for spec in args.configs:
        config_path, label, pretrain = parse_candidate(spec)
        logger.info(f"=== candidate {label} ({config_path}, pretrain={pretrain}) ===")

        cache = os.path.join(args.cache_dir, f"{label}.npz") if args.cache_dir else None
        if cache and os.path.exists(cache):
            d = np.load(cache)
            X, y = d["X"], d["y"]
            timings[label] = (0.0, 0.0)
            logger.info(f"loaded cached features {X.shape} from {cache}")
        else:
            model, loader, num_classes, ann_file, class_map = build_one(config_path, pretrain, logger)
            t0 = time.time()
            X, y = extract_features(
                model, loader, ann_file=ann_file, class_map_file=class_map,
                max_frames=args.max_frames, per_clip_cap=args.per_clip_cap,
                bg_fraction=args.bg_fraction, logger=logger,
            )
            t_extract = time.time() - t0
            del model
            torch.cuda.empty_cache()
            if cache:
                np.savez(cache, X=X, y=y)
            timings[label] = (t_extract, 0.0)

        t1 = time.time()
        scores = {m: score_transferability(X, y, m) for m in args.methods}
        t_score = time.time() - t1
        timings[label] = (timings[label][0], t_score)
        logger.info(
            f"{label}: " + ", ".join(f"{m}={v:.4f}" for m, v in scores.items())
            + f"  [extract {timings[label][0]:.1f}s, score {t_score:.2f}s]"
        )
        rows.append((label, scores))

    # ---- ranking table ----
    print("\n" + "=" * 72)
    print("TRANSFERABILITY RANKING  (higher = better; rank in parens)")
    print("=" * 72)
    header = f"{'backbone':<18}" + "".join(f"{m:>14}" for m in args.methods)
    print(header)
    print("-" * len(header))

    # per-method ranks (1 = best)
    ranks = {m: {} for m in args.methods}
    for m in args.methods:
        order = sorted(rows, key=lambda r: r[1][m], reverse=True)
        for i, (label, _) in enumerate(order):
            ranks[m][label] = i + 1

    for label, scores in rows:
        cells = "".join(f"{scores[m]:>9.3f}({ranks[m][label]})" for m in args.methods)
        print(f"{label:<18}{cells}")

    print("-" * len(header))
    # mean rank across methods = the consensus pick
    consensus = sorted(
        rows, key=lambda r: np.mean([ranks[m][r[0]] for m in args.methods])
    )
    print("\nConsensus (mean rank across methods):")
    for i, (label, _) in enumerate(consensus):
        mr = np.mean([ranks[m][label] for m in args.methods])
        agree = len(set(ranks[m][label] for m in args.methods)) == 1
        flag = "" if agree else "   <- methods disagree"
        print(f"  {i+1}. {label}  (mean rank {mr:.2f}){flag}")
    print("-" * 72)
    print("Timing (per backbone):")
    for label, _ in rows:
        te, ts = timings[label]
        print(f"  {label:<18} extract {te:6.1f}s   score {ts:5.2f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
