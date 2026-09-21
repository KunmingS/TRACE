"""Soft-NMS: the reductions that make a long recording tractable must not move a score.

`softnms_1d` is quadratic in the block it is handed. A sliding-window pass over
a 20-hour video hands it one proposal per frame per window — millions of rows —
so it carries two reductions: a `min_score` cut before the sweep, and a split
into overlap-connected groups. Both claim to be exact, and these tests hold them
to a naive oracle that does the arithmetic the slow, obvious way.
"""
import time

import pytest
import torch

from vtrace.models.postprocess import batched_nms, softnms_1d


def oracle(segs, scores, iou_threshold, sigma, min_score, method, t1, t2):
    """Soft-NMS written the obvious way: pick the max, decay everything else, repeat."""
    rows = [
        {"x1": float(s[0]), "x2": float(s[1]), "score": float(v), "idx": i}
        for i, (s, v) in enumerate(zip(segs, scores))
    ]
    kept = []
    while rows:
        best = max(range(len(rows)), key=lambda k: rows[k]["score"])
        cur = rows.pop(best)
        kept.append(cur)
        area_i = cur["x2"] - cur["x1"] + 1e-6
        survivors = []
        for r in rows:
            inter = max(0.0, min(cur["x2"], r["x2"]) - max(cur["x1"], r["x1"]))
            union = area_i + (r["x2"] - r["x1"] + 1e-6) - inter
            ovr = inter / union
            if method == 0:
                w = 0.0 if ovr >= iou_threshold else 1.0
            elif method == 1:
                w = (1.0 - ovr) if ovr >= iou_threshold else 1.0
            elif method == 2:
                w = pow(2.718281828459045, -(ovr * ovr) / sigma)
            elif method == 3:
                w = pow(2.718281828459045, -(ovr * ovr) / sigma) if ovr >= t1 + t2 * area_i else 1.0
            else:
                w = 1.0
            r["score"] *= w
            if r["score"] >= min_score:
                survivors.append(r)
        rows = survivors
    return kept


def as_triples(dets):
    return sorted((round(float(a), 4), round(float(b), 4), round(float(c), 4))
                  for a, b, c in dets)


def oracle_triples(kept):
    return sorted((round(r["x1"], 4), round(r["x2"], 4), round(r["score"], 4)) for r in kept)


def sparse_bouts(n_bouts=12, per_bout=40, gap=500.0, seed=0):
    """Short bursts of overlapping proposals separated by long silence.

    This is what a behaviour model emits over hours of mostly-idle video, and
    the shape the group split exists for.
    """
    g = torch.Generator().manual_seed(seed)
    starts, scores = [], []
    for b in range(n_bouts):
        base = b * gap
        for k in range(per_bout):
            starts.append(base + k / 60.0)
            scores.append(float(torch.rand(1, generator=g)))
    t = torch.tensor(starts)
    return torch.stack([t, t + 1.0 / 30.0], 1), torch.tensor(scores)


KW = dict(iou_threshold=0.0, sigma=0.5, min_score=0.05, method=2, t1=0.0, t2=0.0)


@pytest.mark.parametrize("method", [0, 1, 2, 3])
@pytest.mark.parametrize("iou_threshold", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("min_score", [0.0, 0.05, 0.4])
def test_matches_naive_oracle(method, iou_threshold, min_score):
    segs, scores = sparse_bouts(n_bouts=4, per_bout=15)
    kw = dict(iou_threshold=iou_threshold, sigma=0.5, min_score=min_score,
              method=method, t1=0.1, t2=0.05)
    dets, _ = softnms_1d(segs.clone(), scores.clone(), **kw)
    assert as_triples(dets) == oracle_triples(oracle(segs, scores, **kw))


def test_a_distant_segment_changes_nothing():
    """Adding a bout hours away must leave the first bout's scores untouched."""
    segs, scores = sparse_bouts(n_bouts=1, per_bout=30)
    alone, _ = softnms_1d(segs.clone(), scores.clone(), **KW)

    far_segs = torch.cat([segs, segs + 9999.0])
    far_scores = torch.cat([scores, scores])
    together, _ = softnms_1d(far_segs, far_scores, **KW)

    kept_near = [row for row in together if float(row[0]) < 9000.0]
    assert as_triples(kept_near) == as_triples(alone)


def test_scores_come_out_descending():
    """Callers slice a top-k off the head, so the order is part of the contract."""
    segs, scores = sparse_bouts(n_bouts=8, per_bout=25)
    dets, _ = softnms_1d(segs, scores, **KW)
    assert len(dets) > 1
    values = [float(row[2]) for row in dets]
    assert values == sorted(values, reverse=True)


def test_scores_come_out_descending_when_nothing_overlaps():
    """Every segment in its own group is the case that skips the merge sort.

    Rows gathered from singleton groups arrive in timeline order, so this is
    where a "the sweep already sorted it" shortcut silently hands the caller
    the lowest-scoring detections as its top-k.
    """
    t = torch.arange(40, dtype=torch.float32) * 100.0
    segs = torch.stack([t, t + 1.0], 1)
    scores = torch.rand(40, generator=torch.Generator().manual_seed(5))
    dets, inds = softnms_1d(segs, scores, **KW)
    values = [float(row[2]) for row in dets]
    assert values == sorted(values, reverse=True)
    assert float(dets[0][2]) == pytest.approx(float(scores.max()))
    assert int(inds[0]) == int(scores.argmax())


def test_indices_point_back_at_the_input_rows():
    """The group split reorders rows; the returned indices must still address the input."""
    segs, scores = sparse_bouts(n_bouts=6, per_bout=20)
    dets, inds = softnms_1d(segs, scores, **KW)
    assert torch.allclose(segs[inds], dets[:, :2], atol=1e-4)


def test_everything_below_min_score_still_returns_the_best():
    """The pre-filter must not turn the sweep's one survivor into an empty result."""
    t = torch.tensor([0.0, 10.0, 20.0])
    segs = torch.stack([t, t + 1.0], 1)
    scores = torch.tensor([0.001, 0.003, 0.002])
    dets, inds = softnms_1d(segs, scores, iou_threshold=0.0, sigma=0.5,
                            min_score=0.05, method=2, t1=0.0, t2=0.0)
    assert len(dets) == 1
    assert int(inds[0]) == 1


def test_vanilla_nms_at_zero_threshold_still_suppresses_disjoint_segments():
    """`ovr >= 0` holds for disjoint pairs, so method 0 may not be group-split.

    Splitting here would leave the far segments at their own scores instead of
    zeroed, which is the one case where independent groups are the wrong model.
    """
    t = torch.tensor([0.0, 500.0, 1000.0])
    segs = torch.stack([t, t + 1.0], 1)
    scores = torch.tensor([0.9, 0.8, 0.7])
    kw = dict(iou_threshold=0.0, sigma=0.5, min_score=0.0, method=0, t1=0.0, t2=0.0)
    dets, _ = softnms_1d(segs, scores, **kw)
    assert as_triples(dets) == oracle_triples(oracle(segs, scores, **kw))
    survivors = [round(float(row[2]), 4) for row in dets if float(row[2]) > 0.0]
    assert survivors == [0.9], "only the top segment should keep a score"


def test_labels_follow_their_segments_through_batched_nms():
    segs, scores = sparse_bouts(n_bouts=5, per_bout=20)
    cls_idxs = (torch.arange(len(scores)) % 3).float()
    out_segs, out_scores, out_cls = batched_nms(
        segs, scores, cls_idxs, iou_threshold=0.0, min_score=0.05,
        max_seg_num=2000, use_soft_nms=True, multiclass=True, sigma=0.5,
        voting_thresh=0.7, method=2)
    for seg, cls in zip(out_segs, out_cls):
        hit = (segs[:, 0] - seg[0]).abs() < 1e-4
        assert cls_idxs[hit].eq(cls).any(), "a detection came back with someone else's label"


def test_a_long_sparse_timeline_does_not_go_quadratic():
    """Guard the reduction itself: 300k proposals over hours used to take minutes.

    The budget is loose on purpose — it is here to catch a return to one
    all-rows-against-all-rows sweep, not to police small regressions.
    """
    segs, scores = sparse_bouts(n_bouts=3000, per_bout=100, gap=30.0)
    assert len(scores) == 300_000
    started = time.perf_counter()
    dets, _ = softnms_1d(segs, scores, **KW)
    elapsed = time.perf_counter() - started
    assert len(dets) > 0
    assert elapsed < 60.0, f"soft-NMS took {elapsed:.1f}s on 300k sparse proposals"


@pytest.mark.parametrize("limit", [1, 3, 50, 10_000])
def test_limit_returns_exactly_the_head_of_an_uncapped_sweep(limit):
    """Stopping early may only skip rows the caller was going to discard."""
    segs, scores = sparse_bouts(n_bouts=6, per_bout=25)
    full, _ = softnms_1d(segs.clone(), scores.clone(), **KW)
    capped, _ = softnms_1d(segs.clone(), scores.clone(), limit=limit, **KW)
    assert as_triples(capped) == as_triples(full[:limit])


def test_one_dense_block_is_bounded_by_the_limit():
    """The split cannot help when every proposal overlaps its neighbour.

    A behaviour that never drops below the cutoff leaves one group spanning the
    whole recording, so the iteration cap is the only thing standing between a
    long video and an N^2 sweep.
    """
    n = 200_000
    t = torch.arange(n, dtype=torch.float32) / 60.0
    segs = torch.stack([t, t + 1.0 / 30.0], 1)
    scores = torch.rand(n, generator=torch.Generator().manual_seed(3)) * 0.5 + 0.5

    started = time.perf_counter()
    dets, _ = softnms_1d(segs, scores, limit=2000, **KW)
    elapsed = time.perf_counter() - started
    assert len(dets) == 2000
    assert elapsed < 60.0, f"a single dense block took {elapsed:.1f}s"


def test_tied_scores_keep_the_earlier_segment():
    """Ties are broken by start time now, not by where a row sat in the input.

    The old sweep's tie-break fell out of array position after its swaps; this
    pins the replacement rule so a change to it is a decision, not an accident.
    Only the *choice* between tied overlapping segments differs from before —
    the decayed value handed to the loser is the same either way.
    """
    later_first = torch.tensor([[1.0, 3.0], [0.0, 2.0]])
    scores = torch.tensor([0.9, 0.9])
    dets, inds = softnms_1d(later_first, scores, **KW)
    assert inds.tolist() == [1, 0]
    assert float(dets[0][0]) == 0.0 and float(dets[0][2]) == pytest.approx(0.9)
    assert float(dets[1][2]) < 0.9

    # Identical segments — the real source of ties, one frame seen by two
    # windows — are indistinguishable whichever copy is kept.
    twins = torch.tensor([[0.03, 0.07], [0.03, 0.07]])
    dets, _ = softnms_1d(twins, torch.tensor([1.0, 1.0]), **KW)
    assert [tuple(round(v, 2) for v in row[:2].tolist()) for row in dets] == [(0.03, 0.07)] * 2
    assert float(dets[0][2]) == pytest.approx(1.0)


def test_lockstep_matches_the_plain_sweep_on_the_same_groups():
    """Groups of the same size can go either route; they must agree exactly."""
    from vtrace.models.postprocess import _softnms_sweep, _softnms_sweep_batched
    g = torch.Generator().manual_seed(9)
    for k in (2, 3, 5, 8):
        for method, iou in ((2, 0.0), (1, 0.3), (0, 0.5), (3, 0.0)):
            starts = torch.sort(torch.rand(k, generator=g) * 2.0).values
            segs = torch.stack([starts, starts + 1.0 + torch.rand(k, generator=g)], 1)
            scores = torch.rand(k, generator=g)
            packed = torch.stack([segs[:, 0], segs[:, 1], scores,
                                  segs[:, 1] - segs[:, 0] + 1e-6,
                                  torch.arange(k, dtype=torch.float32)], 1)
            plain = packed.clone()
            kept = _softnms_sweep(plain, iou, 0.5, 0.05, method, 0.1, 0.05, limit=0)
            lock = _softnms_sweep_batched(packed.clone().unsqueeze(0), iou, 0.5, 0.05,
                                          method, 0.1, 0.05, limit=0)
            assert as_triples(plain[:kept, :3]) == as_triples(lock[:, :3]), (k, method, iou)
            assert sorted(plain[:kept, 4].tolist()) == sorted(lock[:, 4].tolist())
