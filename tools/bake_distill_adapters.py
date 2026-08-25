"""Bake a distilled adapter state_dict into a full model checkpoint.

`tools/train.py` only loads weights via --resume / --init-weights, so an
adapter-only file (e.g. /tank/skm/distill_sim/ckpt/adapter_B_step750.pth, 108
`backbone.model.backbone.blocks.<i>.adapter.*` tensors + 4 `proj.*` tensors from the
distillation head) cannot be handed to it directly. This builds the detector the
config describes, loads the adapter tensors into it, and saves the result as a normal
checkpoint that --init-weights accepts.

    python tools/bake_distill_adapters.py configs/fly_courtship_distillB.py \
        --out /path/init_with_distilled_adapters.pth

The `proj.*` keys have no home in the downstream detector and are dropped. Every
adapter key in the file MUST match a parameter of the built model — a mismatch means
the student architecture and this config disagree, and is a hard error rather than a
silently random-initialised adapter.
"""
import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vtrace.config import Config, DictAction  # noqa: E402
from vtrace.models import build_detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--adapter", default=None, help="override cfg.distill_adapter_ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    class_map = cfg.get("class_map") or cfg.dataset.train.class_map
    with open(class_map) as f:
        classes = [ln.strip() for ln in f if ln.strip()]
    cfg.model.num_classes = len(classes)
    print(f"class_map {class_map} -> num_classes={len(classes)} {classes}")

    model = build_detector(cfg.model)
    model_keys = set(model.state_dict().keys())

    adapter_path = args.adapter or cfg.distill_adapter_ckpt
    raw = torch.load(adapter_path, map_location="cpu")
    raw = raw.get("state_dict", raw)
    adapters = {k: v for k, v in raw.items() if "adapter" in k}
    dropped = sorted(set(raw) - set(adapters))
    print(f"adapter file {adapter_path}: {len(raw)} tensors -> {len(adapters)} adapter, "
          f"dropped {len(dropped)} ({dropped})")

    unmatched = [k for k in adapters if k not in model_keys]
    if unmatched:
        raise SystemExit(
            f"{len(unmatched)} adapter keys have no counterpart in the built model, e.g. "
            f"{unmatched[:5]} — the config and the distilled student disagree."
        )
    shape_bad = [k for k in adapters if model.state_dict()[k].shape != adapters[k].shape]
    if shape_bad:
        raise SystemExit(f"shape mismatch on {len(shape_bad)} adapter tensors, e.g. {shape_bad[:5]}")

    before = {k: model.state_dict()[k].clone() for k in adapters}
    missing, unexpected = model.load_state_dict(adapters, strict=False)
    assert not unexpected, unexpected
    moved = sum(1 for k in adapters if not torch.equal(before[k], model.state_dict()[k]))
    print(f"loaded {len(adapters)} adapter tensors ({moved} differ from the random init)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"epoch": -1, "state_dict": model.state_dict()}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
