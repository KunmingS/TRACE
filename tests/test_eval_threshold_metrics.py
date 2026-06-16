"""Tests for the 3-way stratified split and the background-aware /
threshold-recommending Precision evaluator (GitHub issue #3)."""
import json
from collections import Counter

import pytest

from trace_tad.data_prep import (
    _BG_STRATUM,
    _resolve_split_ratios,
    _stratified_split,
)
from trace_tad.evaluations.precision import Precision


# ── Stratified split ────────────────────────────────────────────────────────

def _make_records():
    recs = []
    recs += [(f"A{i}", {"attack"}) for i in range(200)]
    recs += [(f"B{i}", {"invest"}) for i in range(80)]
    recs += [(f"C{i}", {"mount"}) for i in range(20)]
    recs += [(f"AB{i}", {"attack", "invest"}) for i in range(40)]
    recs += [(f"BG{i}", set()) for i in range(30)]  # background-only
    return recs


def test_resolve_split_ratios_three_way_default():
    tr, va, te = _resolve_split_ratios(0.7, None, None)
    assert (round(tr, 3), round(va, 3), round(te, 3)) == (0.7, 0.15, 0.15)


def test_resolve_split_ratios_backcompat_train_only():
    # Legacy callers passing only train_ratio get the remainder split evenly.
    tr, va, te = _resolve_split_ratios(0.8, None, None)
    assert (round(tr, 3), round(va, 3), round(te, 3)) == (0.8, 0.1, 0.1)


def test_resolve_split_ratios_two_way_when_test_zero():
    tr, va, te = _resolve_split_ratios(0.8, None, 0.0)
    assert te == 0.0 and round(tr + va, 6) == 1.0


def test_stratified_split_keeps_per_class_proportions():
    recs = _make_records()
    tr, va, te = _resolve_split_ratios(0.7, None, None)
    sr = [("train", tr), ("validation", va), ("test", te)]
    asg = _stratified_split(recs, sr, seed=42)

    lab_of = {ck: (ls if ls else {_BG_STRATUM}) for ck, ls in recs}
    counts = {s: Counter() for s in ("train", "validation", "test")}
    for ck, s in asg.items():
        for lab in lab_of[ck]:
            counts[s][lab] += 1

    for lab in ("attack", "invest", "mount", _BG_STRATUM):
        total = sum(counts[s][lab] for s in counts)
        # validation+test share should be ~0.30 of each class (rare classes too)
        held = (counts["validation"][lab] + counts["test"][lab]) / total
        assert abs(held - 0.30) <= 0.05, (lab, held)


def test_stratified_split_is_deterministic():
    recs = _make_records()
    sr = [("train", 0.7), ("validation", 0.15), ("test", 0.15)]
    assert _stratified_split(recs, sr, seed=42) == _stratified_split(recs, sr, seed=42)


def test_stratified_split_two_way():
    recs = _make_records()
    asg = _stratified_split(recs, [("train", 0.8), ("validation", 0.2)], seed=42)
    assert set(asg.values()) == {"train", "validation"}


# ── Background-aware mAP + threshold recommendation ──────────────────────────

@pytest.fixture
def gt_file(tmp_path):
    gt = {"database": {"v1": {
        "subset": "validation", "duration": 100 / 30.0, "frame": 100,
        "annotations": [
            {"label": "attack", "segment": [10 / 30.0, 20 / 30.0], "frame_segment": [10, 20]},
            {"label": "attack", "segment": [60 / 30.0, 70 / 30.0], "frame_segment": [60, 70]},
        ],
    }}}
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(gt))
    return str(p)


def _evaluate(gt_file, pred, **kw):
    ev = Precision(gt_file, pred, "validation", tiou_thresholds=[0.5],
                   eval_fps=30.0, thread=1, **kw)
    return ev, ev.evaluate()


def test_bg_aware_map_penalizes_overprediction(gt_file):
    # A model that fires 'attack' across almost the whole clip: high recall,
    # many false positives on background frames -> background-aware mAP is low.
    over = {"results": {"v1": [
        {"segment": [5 / 30.0, 95 / 30.0], "label": "attack", "score": 0.9}]}}
    _, m = _evaluate(gt_file, over)
    assert m["mAP"] < 0.6
    # Only one metric is reported; the legacy exclude-empty value is gone.
    assert "mAP_nonempty" not in m


def test_good_model_scores_high_and_recommends_threshold(gt_file):
    good = {"results": {"v1": [
        {"segment": [10 / 30.0, 20 / 30.0], "label": "attack", "score": 0.9},
        {"segment": [60 / 30.0, 70 / 30.0], "label": "attack", "score": 0.85},
        {"segment": [40 / 30.0, 45 / 30.0], "label": "attack", "score": 0.2},  # weak FP
    ]}}
    _, m = _evaluate(gt_file, good)
    assert m["mAP"] > 0.95
    rec = m["recommended_thresholds"]
    assert 0.0 < rec["global"]["threshold"] < 1.0
    assert "attack" in rec["per_class"]
    # P/R/F1 are reported at the chosen threshold (not the ~0 cutoff).
    assert m["eval_threshold"] == rec["global"]["threshold"]


def test_explicit_score_threshold_is_respected(gt_file):
    good = {"results": {"v1": [
        {"segment": [10 / 30.0, 20 / 30.0], "label": "attack", "score": 0.9}]}}
    _, m = _evaluate(gt_file, good, score_threshold=0.5)
    assert m["eval_threshold"] == 0.5


# ── Predict-time threshold handoff (per-class) ───────────────────────────────

def test_filter_predictions_per_class_thresholds():
    from trace_tad.video_annotation import filter_predictions

    spec = {"global": 0.3, "per_class": {"attack": 0.5, "mount": None}}
    preds = {"v": [
        {"label": "attack", "score": 0.45},   # below per-class 0.5 -> dropped
        {"label": "attack", "score": 0.55},   # kept
        {"label": "invest", "score": 0.35},   # no per-class entry -> global 0.3 -> kept
        {"label": "invest", "score": 0.20},   # below global -> dropped
        {"label": "mount", "score": 0.31},    # per-class None -> global 0.3 -> kept
    ]}
    kept = [(d["label"], d["score"]) for d in filter_predictions(preds, spec)["v"]]
    assert kept == [("attack", 0.55), ("invest", 0.35), ("mount", 0.31)]


def test_filter_predictions_scalar_backcompat():
    from trace_tad.video_annotation import filter_predictions

    preds = {"v": [{"label": "a", "score": 0.4}, {"label": "b", "score": 0.6}]}
    kept = [(d["label"], d["score"]) for d in filter_predictions(preds, 0.5)["v"]]
    assert kept == [("b", 0.6)]


def test_threshold_display_handles_per_class_and_scalar():
    from trace_tad.video_annotation import _threshold_display

    assert _threshold_display(0.3) == "0.30"
    assert _threshold_display({"global": 0.3, "per_class": {}}).startswith("per-class")
