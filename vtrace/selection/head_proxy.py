"""Tier-2 cheap proxy training for per-frame HEAD selection.

Transferability metrics cannot rank heads (frozen backbone -> same features ->
same score), so we proxy-train each candidate head on **cached frozen-backbone
sequences** and rank by validation frame-mAP. The backbone forward is done once
(adapter OFF, head-independent); only the tiny head trains, so a whole head menu
ranks in minutes. See ``model-selection-recipe.md (research notes, archived)`` (Tier 2).

This is a *proxy*: single-scale, no projection/neck, no localization head, adapter OFF
— so absolute mAP differs from the full pipeline. The question it answers is
whether the cheap proxy reproduces the head **ranking** from the full bake-off.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..models.heads import build_frame_cls_head
from .extract import _meta_get, load_gt


@torch.no_grad()
def extract_sequences(
    model, loader, *, ann_file, class_map_file, num_classes,
    max_windows=600, device="cuda", logger=None,
):
    """Frozen backbone over a sliding-window loader -> list of per-window
    ``(feat[C,T] fp16 cpu, label[T] int64, valid[T] bool)``. bg = ``num_classes``.
    """
    model.eval()
    backbone = model.backbone
    gt, classes = load_gt(ann_file, class_map_file)
    bg = num_classes
    seqs = []

    for data_dict in loader:
        inputs = data_dict["inputs"].to(device, non_blocking=True)
        if inputs.dim() == 6:
            inputs = inputs[:, 0:1].contiguous()
        elif inputs.dim() == 5:
            inputs = inputs.unsqueeze(1).contiguous()
        feats = backbone(inputs)                       # [B, C, T]
        B, C, T = feats.shape
        feats = feats.float().cpu()
        metas = data_dict.get("metas", None)
        masks = data_dict.get("masks", None)
        if metas is None:
            continue
        for b in range(B):
            meta = metas[b]
            vname = _meta_get(meta, "video_name")
            wstart = _meta_get(meta, "window_start_frame", 0) or 0
            if vname is None or vname not in gt:
                continue
            y = np.full(T, bg, dtype=np.int64)
            for (s, e, lab) in gt[vname]:
                a = max(int(np.floor(s - wstart)), 0)
                z = min(int(np.ceil(e - wstart)), T)
                if z > a:
                    y[a:z] = lab
            valid = np.ones(T, dtype=bool)
            if masks is not None:
                m = masks[b]
                m = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
                m = m.reshape(-1).astype(bool)
                if m.shape[0] == T:
                    valid = m
            seqs.append((feats[b].half(), torch.from_numpy(y), torch.from_numpy(valid)))
        if len(seqs) >= max_windows:
            break
    if logger:
        logger.info(f"extracted {len(seqs)} windows, dim {C}, T {T}")
    return seqs[:max_windows], classes


def _class_weights(seqs, num_classes, device):
    counts = np.zeros(num_classes + 1)
    for _, y, valid in seqs:
        yv = y[valid].numpy()
        c, n = np.unique(yv, return_counts=True)
        for ci, ni in zip(c, n):
            counts[ci] += ni
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))         # inv_freq_sqrt (bake-off recipe)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def _frame_map(probs, labels, num_classes):
    """sklearn non-interpolated AP per real class, mean over classes that occur
    (mAP_all_frames convention: bg excluded, treated as negative)."""
    from sklearn.metrics import average_precision_score
    aps = {}
    for c in range(num_classes):
        yt = (labels == c).astype(np.int32)
        if yt.sum() == 0:
            continue
        aps[c] = float(average_precision_score(yt, probs[:, c]))
    mean = float(np.mean(list(aps.values()))) if aps else 0.0
    return mean, aps


def train_eval_head(
    head_type, train_seqs, val_seqs, *, num_classes, in_ch=768, feat_ch=512,
    routing=None, epochs=12, lr=1e-3, batch_size=16, device="cuda",
    seed=0, logger=None,
):
    """Proxy-train one head on cached sequences; return (val_frame_mAP, per_class)."""
    torch.manual_seed(seed)
    head = build_frame_cls_head(
        head_type, in_ch=in_ch, feat_ch=feat_ch, out_ch=num_classes + 1,
        num_layers=2, routing=routing,
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    w = _class_weights(train_seqs, num_classes, device)
    n = len(train_seqs)
    rng = np.random.RandomState(seed)

    head.train()
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            x = torch.stack([train_seqs[j][0] for j in idx]).float().to(device)   # [B,C,T]
            y = torch.stack([train_seqs[j][1] for j in idx]).to(device)            # [B,T]
            v = torch.stack([train_seqs[j][2] for j in idx]).to(device)            # [B,T]
            logits = head(x)                                                       # [B,C+1,T]
            ce = F.cross_entropy(logits, y, weight=w, reduction="none")            # [B,T]
            loss = (ce * v).sum() / v.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for i in range(0, len(val_seqs), batch_size):
            chunk = val_seqs[i:i + batch_size]
            x = torch.stack([c[0] for c in chunk]).float().to(device)
            logits = head(x)                                                       # [B,C+1,T]
            p = torch.softmax(logits, dim=1).cpu().numpy()                         # [B,C+1,T]
            for k, (_, y, v) in enumerate(chunk):
                vm = v.numpy()
                all_p.append(p[k].T[vm])                                           # [Tv, C+1]
                all_y.append(y.numpy()[vm])
    probs = np.concatenate(all_p, 0)
    labels = np.concatenate(all_y, 0)
    mean, per_class = _frame_map(probs, labels, num_classes)
    if logger:
        logger.info(f"  {head_type}: frame_mAP={mean*100:.2f} per_class={{ {', '.join(f'{c}:{v*100:.2f}' for c,v in per_class.items())} }}")
    return mean, per_class


def successive_halving_routing(
    train_seqs, val_seqs, *, num_classes, candidates,
    in_ch=768, feat_ch=512, budget=(4, 8, 16), keep_frac=0.5,
    default=None, margin=1.0, seeds=(0,), max_rounds=None, device="cuda",
    logger=None,
):
    """Per-class successive-halving search for a ``RoutedFrameHead`` routing map.

    Discovers the routing instead of hand-assigning it: every real class keeps a
    set of candidate *specialist* heads. Each round, the ``default`` head plus
    every still-alive specialist is proxy-trained on the cached frozen-backbone
    sequences at that round's epoch budget (rising Hyperband-style); then, per
    class, the lowest-AP specialists are eliminated (keep the top ``keep_frac``,
    always drop >=1). A head that survives for no class is dropped entirely -- the
    head menu "dies off" as validation rounds progress. Converges when every
    class has <=1 surviving specialist.

    The final routing uses the same safety valve as ``select_head --derive-routing``:
    a class is handed to its surviving specialist only if it beats ``default`` on
    that class by >= ``margin`` AP -- with both compared at the *final* round's
    budget, so the numbers are apples-to-apples; otherwise the class stays on
    ``default``. ``default`` is never eliminated, so it is always a fair yardstick.

    Returns ``dict(routing=<config-ready dict>, default=<head>, rounds=[...],
    per_class=<final per-class AP by head>, decisions=[...], candidates=[...])``.
    """
    log = (logger.info if logger else (lambda *a, **k: None))
    cand = [h for h in dict.fromkeys(candidates) if h != "routed"]  # de-dup, drop 'routed'
    assert len(cand) >= 2, "successive-halving needs >=2 candidate heads"
    budget = list(budget) or [8]

    def _avg_eval(ht, epochs):
        ms, pcs = [], []
        for sd in seeds:
            m, pc = train_eval_head(ht, train_seqs, val_seqs, num_classes=num_classes,
                                    in_ch=in_ch, feat_ch=feat_ch, epochs=epochs,
                                    seed=sd, device=device)
            ms.append(m); pcs.append(pc)
        keys = set().union(*[set(pc) for pc in pcs]) if pcs else set()
        per_class = {c: float(np.mean([pc.get(c, 0.0) for pc in pcs])) for c in keys}
        return float(np.mean(ms)), per_class

    # specialists[c]: surviving specialist heads for class c (default is excluded
    # from this set -- it is always available and never eliminated).
    specialists = {c: None for c in range(num_classes)}    # filled after default known
    last_pc = {}                                            # head -> latest per-class AP

    def _eliminate(round_pc):
        elim = {}
        for c in range(num_classes):
            spec = specialists[c]
            if spec is None or len(spec) <= 1:
                continue
            ranked = sorted(spec, key=lambda h: round_pc.get(h, {}).get(c, 0.0), reverse=True)
            k = max(1, math.ceil(keep_frac * len(ranked)))
            if k >= len(ranked):
                k = len(ranked) - 1
            for h in ranked[k:]:
                spec.discard(h)
            if ranked[k:]:
                elim[c] = ranked[k:]
        return elim

    # ---- round 0: train every candidate, pick/confirm default, first cut ----
    rounds = []
    ep0 = budget[0]
    scores = {ht: _avg_eval(ht, ep0) for ht in cand}
    means0 = {ht: scores[ht][0] for ht in cand}
    for ht in cand:
        last_pc[ht] = scores[ht][1]
    if default is None:
        default = max(means0, key=means0.get)
    log(f"[SHA] round 0 ({ep0}ep) default(anchor)='{default}' "
        f"means={{{', '.join(f'{h}:{means0[h]*100:.1f}' for h in cand)}}}")
    for c in range(num_classes):
        specialists[c] = set(h for h in cand if h != default)
    elim0 = _eliminate({h: scores[h][1] for h in cand})
    rounds.append(dict(round=0, budget=ep0, means=means0,
                       per_class={h: scores[h][1] for h in cand}, eliminated=elim0))

    # ---- successive-halving rounds at rising budget ----
    max_rounds = max_rounds or (len(cand) + 3)
    r = 1
    while any(len(specialists[c]) > 1 for c in range(num_classes)) and r < max_rounds:
        ep = budget[min(r, len(budget) - 1)]
        alive = set().union(*specialists.values()) | {default}
        active = [h for h in cand if h in alive]                # stable order
        rscore = {ht: _avg_eval(ht, ep) for ht in active}
        for ht in active:
            last_pc[ht] = rscore[ht][1]
        round_pc = {ht: rscore[ht][1] for ht in active}
        elim = _eliminate(round_pc)
        rounds.append(dict(round=r, budget=ep,
                           means={h: rscore[h][0] for h in active},
                           per_class=round_pc, eliminated=elim))
        log(f"[SHA] round {r} ({ep}ep) active={active} "
            f"eliminated={{{', '.join(f'{c}:{v}' for c, v in elim.items())}}}")
        r += 1

    # ---- assemble routing: surviving specialist vs default (margin valve) ----
    routing = {"default": default}
    decisions = []
    for c in range(num_classes):
        spec = sorted(specialists[c], key=lambda h: last_pc.get(h, {}).get(c, 0.0), reverse=True)
        d_ap = last_pc.get(default, {}).get(c, 0.0)
        if spec:
            w = spec[0]
            gain = last_pc.get(w, {}).get(c, 0.0) - d_ap
            if gain * 100 >= margin:
                routing.setdefault(w, []).append(c)
                decisions.append(dict(cls=c, head=w, gain=gain, routed=True))
                continue
        decisions.append(dict(cls=c, head=default, gain=0.0, routed=False))

    return dict(routing=routing, default=default, rounds=rounds,
                per_class=last_pc, decisions=decisions, candidates=cand)
