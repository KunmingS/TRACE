"""Extract head-independent frozen-backbone features + per-frame labels.

The transferability metrics need ``(X[N,D], y[N])`` where each row is one
temporal location's pooled backbone feature and its behavior-class label. We
take the **raw backbone output** ``[B, C, T]`` (the wrapper's pooled output, NOT
post-projection/neck which are randomly initialised in a fresh model) and
disable the adapter (``adapter_index=[]``) so the representation is purely the
pretrained backbone — the only feature set independent of both the head and
downstream training (see ``model-selection-recipe.md (research notes, archived)``).

Labels come straight from the annotation JSON, rasterised onto each sliding
window using ``metas`` (``video_name`` + ``window_start_frame``). This sidesteps
the inference pipeline not collecting ``gt_segments`` and works for sliding
windows over **original videos** (the protocol used here), where some windows
have no behavior (pure background).
"""

from __future__ import annotations

import json

import numpy as np
import torch


def load_gt(ann_file: str, class_map_file: str):
    """video_name -> list of (start_frame, end_frame, class_idx) for scored
    classes only (skips 'Ambiguous' and any label not in the class map)."""
    with open(class_map_file) as f:
        classes = [ln.strip() for ln in f if ln.strip()]
    cls_idx = {c: i for i, c in enumerate(classes)}
    with open(ann_file) as f:
        db = json.load(f)
    db = db.get("database", db)

    gt = {}
    for name, info in db.items():
        segs = []
        dur = info.get("duration", None)
        frm = info.get("frame", None)
        for a in info.get("annotations", []):
            lab = a.get("label")
            if lab is None or lab == "Ambiguous" or lab not in cls_idx:
                continue
            fs = a.get("frame_segment")
            if fs is None:
                if dur and frm:
                    fs = [a["segment"][0] / dur * frm, a["segment"][1] / dur * frm]
                else:
                    continue
            segs.append((float(fs[0]), float(fs[1]), cls_idx[lab]))
        gt[name] = segs
    return gt, classes


def _meta_get(meta, key, default=None):
    v = meta.get(key, default) if isinstance(meta, dict) else default
    if torch.is_tensor(v):
        v = v.item()
    return v


@torch.no_grad()
def extract_features(
    model,
    loader,
    *,
    ann_file: str,
    class_map_file: str,
    bg_label: int = -1,
    max_frames: int = 40000,
    per_clip_cap: int = 256,
    bg_fraction: float = 0.5,
    device: str = "cuda",
    logger=None,
):
    """Run the frozen backbone over a sliding-window loader and collect pooled
    per-frame features + per-window rasterised labels.

    Returns (X[N, D] float32, y[N] int64).
    """
    model.eval()
    backbone = model.backbone
    gt, classes = load_gt(ann_file, class_map_file)
    if logger:
        logger.info(f"loaded gt for {len(gt)} videos, {len(classes)} classes: {classes}")

    feats_all, labels_all = [], []
    n_collected = 0
    rng = np.random.RandomState(0)
    n_no_meta = 0

    for data_dict in loader:
        inputs = data_dict["inputs"].to(device, non_blocking=True)
        if inputs.dim() == 6:
            inputs = inputs[:, 0:1].contiguous()          # uniform stream only
        elif inputs.dim() == 5:
            inputs = inputs.unsqueeze(1).contiguous()      # add num_segs=1

        feats = backbone(inputs)                           # [B, C, T]
        if feats.dim() != 3:
            raise RuntimeError(f"expected [B,C,T] backbone output, got {tuple(feats.shape)}")
        B, C, T = feats.shape
        feats = feats.float().cpu().numpy()

        metas = data_dict.get("metas", None)
        if metas is None:
            n_no_meta += B
            continue
        masks = data_dict.get("masks", None)

        for b in range(B):
            meta = metas[b]
            vname = _meta_get(meta, "video_name")
            wstart = _meta_get(meta, "window_start_frame", 0) or 0
            if vname is None or vname not in gt:
                continue

            # window covers absolute frames [wstart, wstart+T); feature axis is
            # 1:1 with frames (snippet_stride=1, output interpolated to window_size).
            y = np.full(T, bg_label, dtype=np.int64)
            for (s, e, lab) in gt[vname]:
                a = int(np.floor(s - wstart))
                z = int(np.ceil(e - wstart))
                a = max(a, 0)
                z = min(z, T)
                if z > a:
                    y[a:z] = lab

            fb = feats[b].T                                # [T, C]

            # valid = real (non-padded) frames; SingleMouse clips < 768 get
            # padded to the window, so exclude the padding from sampling.
            valid = np.ones(T, dtype=bool)
            if masks is not None:
                m = masks[b]
                m = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
                m = m.reshape(-1).astype(bool)
                if m.shape[0] == T:
                    valid = m

            fg_idx = np.where((y != bg_label) & valid)[0]
            bg_idx = np.where((y == bg_label) & valid)[0]
            n_fg = int(round(per_clip_cap * (1.0 - bg_fraction)))
            n_bg = per_clip_cap - n_fg
            pick = []
            if len(fg_idx):
                pick.append(rng.choice(fg_idx, min(n_fg, len(fg_idx)), replace=False))
            if len(bg_idx):
                pick.append(rng.choice(bg_idx, min(n_bg, len(bg_idx)), replace=False))
            if not pick:
                continue
            sel = np.concatenate(pick)
            feats_all.append(fb[sel])
            labels_all.append(y[sel])
            n_collected += len(sel)

        if n_collected >= max_frames:
            break

    if n_no_meta and logger:
        logger.info(f"WARNING: {n_no_meta} samples had no metas (skipped)")
    if not feats_all:
        raise RuntimeError("no features collected — check ann_file/class_map and that metas carry video_name")

    X = np.concatenate(feats_all, axis=0).astype(np.float32)
    Y = np.concatenate(labels_all, axis=0).astype(np.int64)
    if len(X) > max_frames:
        idx = rng.choice(len(X), max_frames, replace=False)
        X, Y = X[idx], Y[idx]
    if logger:
        cls, cnt = np.unique(Y, return_counts=True)
        logger.info(f"extracted X={X.shape} label dist={dict(zip(cls.tolist(), cnt.tolist()))}")
    return X, Y
