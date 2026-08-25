"""Tier-2 fast per-frame HEAD selection by cheap proxy training.

Caches frozen-backbone sequences ONCE (adapter OFF, head-independent), then
proxy-trains each candidate head on them and ranks by validation frame-mAP. The
backbone forward dominates; the heads are tiny, so the whole menu ranks in
minutes. See ``model-selection-recipe.md (research notes, archived)`` (Tier 2).

Usage (CalMS21, K710 backbone, full head menu):
  python tools/select_head.py \
    --config configs/calms21_vitB_fullseq_select.py \
    --pretrain /mnt/ssd2/skm/TRACE/pretrained/vitB_videomaev2_k710.pth \
    --train-data-path /mnt/ssd2/skm/TRACE/data/calms21/task1_videos_mp4/train \
    --val-data-path   /mnt/ssd2/skm/TRACE/data/calms21/task1_videos_mp4/test \
    --heads conv tcn attn asformer ssm dyfadet routed --routed-tcn 2
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtrace.config import Config
from vtrace.datasets import build_dataset, build_dataloader
from vtrace.models import build_detector
from vtrace.selection.head_proxy import extract_sequences, train_eval_head
from vtrace.utils import setup_logger


def build_split(cfg, subset, data_path, logger):
    split = copy.deepcopy(cfg.dataset.test)         # sliding, test_mode=True (metas, all windows)
    split["subset_name"] = subset
    if data_path:
        split["data_path"] = data_path
    ds = build_dataset(split, default_args=dict(logger=logger))
    loader = build_dataloader(ds, shuffle=False, drop_last=False, **cfg.solver.test)
    return ds, loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pretrain", required=True)
    ap.add_argument("--heads", nargs="+",
                    default=["conv", "tcn", "attn", "asformer", "ssm", "dyfadet", "routed"])
    ap.add_argument("--routed-tcn", type=int, nargs="+", default=None,
                    help="class idxs routed to tcn in 'routed' (rest -> attn). "
                         "CalMS21 mount=2; SingleMouse eating=1")
    ap.add_argument("--train-data-path", default=None)
    ap.add_argument("--val-data-path", default=None)
    ap.add_argument("--derive-routing", action="store_true",
                    help="after training single heads, auto-derive a routing map "
                         "(per-behavior winner) and print it ready to paste into a routed config")
    ap.add_argument("--routing-default", default=None,
                    help="anchor the routed default to this head (recommended: a trusted "
                         "generalist like 'attn'). If unset, uses the best-mean single head "
                         "(can collapse if the proxy inflates one head).")
    ap.add_argument("--routing-margin", type=float, default=1.0,
                    help="route a behavior to a specialist only if it beats the default on "
                         "that class by >= this many AP points")
    ap.add_argument("--sha-routing", action="store_true",
                    help="per-class SUCCESSIVE-HALVING auto-routing: instead of a single "
                         "argmax pass (--derive-routing), gradually eliminate heads over "
                         "rising-budget rounds (by per-class val AP) and emit the routing")
    ap.add_argument("--sha-budget", type=int, nargs="+", default=[4, 8, 16],
                    help="epochs per SHA round (rising Hyperband budget; last value reused "
                         "if more rounds are needed)")
    ap.add_argument("--sha-keep-frac", type=float, default=0.5,
                    help="fraction of each class's surviving specialists kept per round "
                         "(the rest are eliminated; always drops >=1)")
    ap.add_argument("--save-routing", default=None,
                    help="write the discovered routing (+ class names) to this JSON path")
    ap.add_argument("--max-train-windows", type=int, default=600)
    ap.add_argument("--max-val-windows", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    logger = setup_logger("SelectHead", save_dir=None)
    cfg = Config.fromfile(args.config)
    cfg.model.backbone["adapter_index"] = []                      # head-independent features
    cfg.model.backbone.setdefault("custom", {})["pretrain"] = args.pretrain

    ann_file = cfg.dataset.test["ann_file"]
    class_map = cfg.dataset.test["class_map"]

    # ---- cache frozen sequences once (train + val) ----
    cache = None
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache = os.path.join(args.cache_dir, "seqs.pt")
    if cache and os.path.exists(cache):
        blob = torch.load(cache)
        train_seqs, val_seqs, num_classes = blob["train"], blob["val"], blob["num_classes"]
        logger.info(f"loaded cached sequences: {len(train_seqs)} train, {len(val_seqs)} val")
        t_extract = 0.0
    else:
        model = build_detector(cfg.model).cuda().eval()
        _, train_loader = build_split(cfg, "train", args.train_data_path, logger)
        val_ds, val_loader = build_split(cfg, "validation", args.val_data_path, logger)
        num_classes = len(val_ds.class_map)
        t0 = time.time()
        train_seqs, _ = extract_sequences(model, train_loader, ann_file=ann_file,
            class_map_file=class_map, num_classes=num_classes,
            max_windows=args.max_train_windows, logger=logger)
        val_seqs, _ = extract_sequences(model, val_loader, ann_file=ann_file,
            class_map_file=class_map, num_classes=num_classes,
            max_windows=args.max_val_windows, logger=logger)
        t_extract = time.time() - t0
        del model
        torch.cuda.empty_cache()
        if cache:
            torch.save(dict(train=train_seqs, val=val_seqs, num_classes=num_classes), cache)
    logger.info(f"sequence extraction: {t_extract:.1f}s ({len(train_seqs)} train / {len(val_seqs)} val, {num_classes} classes)")

    with open(class_map) as f:
        classes = [ln.strip() for ln in f if ln.strip()]

    # ---- successive-halving auto-routing (gradual head elimination over rounds) ----
    if args.sha_routing:
        from vtrace.selection.head_proxy import successive_halving_routing
        cand = [h for h in args.heads if h != "routed"]
        res = successive_halving_routing(
            train_seqs, val_seqs, num_classes=num_classes, candidates=cand,
            in_ch=train_seqs[0][0].shape[0], budget=args.sha_budget,
            keep_frac=args.sha_keep_frac, default=args.routing_default,
            margin=args.routing_margin, seeds=args.seeds, logger=logger,
        )
        print("\n" + "=" * 78)
        print("SUCCESSIVE-HALVING AUTO-ROUTING (per-class, frozen-backbone proxy)")
        print(f"candidates={res['candidates']}  default='{res['default']}'  "
              f"budget={args.sha_budget}ep  keep_frac={args.sha_keep_frac}  margin={args.routing_margin}AP")
        print("=" * 78)
        for rd in res["rounds"]:
            means = "  ".join(f"{h}:{m*100:.1f}" for h, m in rd["means"].items())
            elim = "; ".join(f"{classes[c]}<-drop[{','.join(d)}]"
                             for c, d in rd["eliminated"].items()) or "(none)"
            print(f"round {rd['round']} ({rd['budget']:>3}ep)  mean[{means}]")
            print(f"            eliminated: {elim}")
        print("-" * 78)
        for d in res["decisions"]:
            tag = f"ROUTE -> {d['head']} (+{d['gain']*100:.2f} AP)" if d["routed"] \
                else f"kept on default '{res['default']}'"
            print(f"  {classes[d['cls']]:<16} {tag}")
        print("-" * 78)
        print(f"  aux_frame_cls.routing = {res['routing']}")
        print(f"  (class idxs: {dict(enumerate(classes))})")
        print("=" * 78)
        if args.save_routing:
            with open(args.save_routing, "w") as f:
                json.dump(dict(routing=res["routing"], default=res["default"],
                               classes=classes, decisions=res["decisions"]), f, indent=2)
            print(f"[saved] {args.save_routing}")
        return

    routing = None
    if args.routed_tcn is not None:
        routing = dict(default="attn", tcn=list(args.routed_tcn))

    # ---- proxy-train each head ----
    results = {}
    timings = {}
    for ht in args.heads:
        r = routing if ht == "routed" else None
        if ht == "routed" and routing is None:
            logger.info("skipping 'routed' (no --routed-tcn given)"); continue
        t0 = time.time()
        maps, pcs = [], []
        for sd in args.seeds:
            m, pc = train_eval_head(ht, train_seqs, val_seqs, num_classes=num_classes,
                in_ch=train_seqs[0][0].shape[0], routing=r, epochs=args.epochs,
                seed=sd, logger=logger)
            maps.append(m); pcs.append(pc)
        timings[ht] = time.time() - t0
        results[ht] = (float(np.mean(maps)), float(np.std(maps)), pcs[0])

    # ---- ranking table ----
    with open(class_map) as f:
        classes = [ln.strip() for ln in f if ln.strip()]
    order = sorted(results.items(), key=lambda kv: kv[1][0], reverse=True)
    print("\n" + "=" * 78)
    print(f"HEAD PROXY RANKING (val frame-mAP %, frozen backbone, {len(args.seeds)} seed(s))")
    print("=" * 78)
    print(f"{'rank':<5}{'head':<10}{'frame_mAP':>11}{'std':>7}   per-class AP")
    print("-" * 78)
    for i, (ht, (m, s, pc)) in enumerate(order):
        pcs = "  ".join(f"{classes[c]}:{v*100:.1f}" for c, v in pc.items())
        print(f"{i+1:<5}{ht:<10}{m*100:>11.2f}{s*100:>7.2f}   {pcs}")
    print("-" * 78)
    print("Timing (proxy train+eval per head):")
    for ht in args.heads:
        if ht in timings:
            print(f"  {ht:<10} {timings[ht]:6.1f}s")
    print("=" * 78)

    # ---- auto-derive routing from per-behavior winners (single heads only) ----
    if args.derive_routing:
        singles = {ht: pc for ht, (_, _, pc) in results.items() if ht != "routed"}
        if not singles:
            print("\n[derive-routing] no single-head results to derive from"); return
        means = {ht: results[ht][0] for ht in singles}
        default = args.routing_default or max(means, key=means.get)
        routing = {"default": default}
        print(f"\n{'=' * 78}\nDERIVED ROUTING (default='{default}', margin={args.routing_margin} AP)\n{'=' * 78}")
        for c in range(num_classes):
            best = max(singles, key=lambda h: singles[h].get(c, 0.0))
            gain = singles[best].get(c, 0.0) - singles[default].get(c, 0.0)
            if best != default and gain * 100 >= args.routing_margin:
                routing.setdefault(best, []).append(c)
                print(f"  {classes[c]:<16} -> {best:<9} (+{gain*100:.2f} AP over default '{default}')  ROUTE")
            else:
                cand = best if best != default else sorted(singles, key=lambda h: -singles[h].get(c, 0))[1]
                print(f"  {classes[c]:<16} -> {default:<9} (kept; {cand} only +{(singles[cand].get(c,0)-singles[default].get(c,0))*100:.2f})")
        # render as a config-ready routing dict (head -> list of class idxs)
        rd = {"default": default}
        for ht, cls in routing.items():
            if ht != "default":
                rd[ht] = cls
        print(f"\n  aux_frame_cls.routing = {rd}")
        print(f"  (class idxs: {dict(enumerate(classes))})")
        print("=" * 78)


if __name__ == "__main__":
    main()
