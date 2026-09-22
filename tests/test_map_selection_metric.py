"""mAP has to be able to tell two models apart, because it picks the checkpoint.

`tools/train.py` keeps the epoch with the highest validation mAP. A metric that
is pinned to the same value whatever the model says turns that into a coin toss
— and worse than a coin toss, since the comparison is a strict `>`, so the
earliest scored epoch wins every tie and the least-trained model is published.

That is what happened on a one-behaviour dataset: `map_frame_filter='nonempty_gt'`
keeps only frames carrying a label, every one of those is a positive for the only
class, and AP over an all-positive column is 1.0 by arithmetic.
"""
import numpy as np
import pytest

from vtrace.evaluations.precision import Precision


def evaluator(map_frame_filter="all", ap_mode="sklearn"):
    """A Precision with the plumbing bypassed — these tests drive the maths directly."""
    p = Precision.__new__(Precision)
    p.map_frame_filter = map_frame_filter
    p.ap_mode = ap_mode
    return p


def frames(scores, truth):
    """(scores, gt_binary, nonempty, labels) as `_build_frame_matrices` returns it."""
    scores = np.asarray(scores, dtype=np.float32).reshape(len(scores), -1)
    truth = np.asarray(truth, dtype=np.int8).reshape(scores.shape)
    return scores, truth, truth.any(axis=1), [f"c{i}" for i in range(scores.shape[1])]


@pytest.mark.parametrize("scores, name", [
    (np.linspace(0, 1, 200), "ranked well"),
    (np.linspace(1, 0, 200), "ranked backwards"),
    (np.zeros(200), "no signal at all"),
])
def test_all_positive_column_has_no_measurable_ap(scores, name):
    """The three models below are not equally good; AP cannot say so."""
    p = evaluator()
    ap = p.compute_average_precision(np.asarray(scores), np.ones(200, dtype=np.int8))
    assert ap == pytest.approx(1.0), f"{name}: AP is pinned, which is the premise"

    p._frame_matrices = frames(scores, np.ones(200))
    out = p.compute_frame_based_mAP()
    assert out["mAP"] is None, f"{name}: an unmeasurable split must not report a score"
    assert out["undefined_AP"] == ["c0"]
    assert out["per_class_AP"]["c0"] is None


def test_background_frames_make_the_single_class_measurable():
    """Same predictions, background frames included: now the ranking matters."""
    rng = np.random.default_rng(0)
    truth = np.zeros(2000, dtype=np.int8)
    truth[:100] = 1                       # 5% positives, like a behaviour corpus

    good = np.where(truth == 1, 0.9, 0.1) + rng.normal(0, 0.01, 2000)
    bad = rng.random(2000)
    backwards = np.where(truth == 1, 0.1, 0.9) + rng.normal(0, 0.01, 2000)

    aps = []
    for scores in (good, bad, backwards):
        p = evaluator()
        p._frame_matrices = frames(scores, truth)
        out = p.compute_frame_based_mAP()
        assert out["mAP"] is not None and out["undefined_AP"] == []
        aps.append(out["mAP"])

    good_ap, bad_ap, backwards_ap = aps
    assert good_ap > bad_ap > backwards_ap, aps
    assert good_ap > 0.9 and backwards_ap < 0.1


def test_nonempty_gt_is_what_hid_the_problem():
    """With a second class present, `nonempty_gt` measures something; with one, nothing."""
    # two classes, disjoint: each class's frames are negatives for the other
    truth = np.zeros((400, 2), dtype=np.int8)
    truth[:200, 0] = 1
    truth[200:, 1] = 1
    scores = np.zeros((400, 2), dtype=np.float32)
    scores[:200, 0] = 0.9
    scores[200:, 1] = 0.9

    p = evaluator(map_frame_filter="nonempty_gt", ap_mode="sklearn")
    p._frame_matrices = (scores, truth, truth.any(axis=1), ["a", "b"])
    out = p.compute_frame_based_mAP()
    assert out["mAP"] == pytest.approx(1.0)
    assert out["undefined_AP"] == []      # both classes had negatives to rank against

    # one class: every frame `nonempty_gt` keeps is a positive
    p = evaluator(map_frame_filter="nonempty_gt", ap_mode="sklearn")
    one = np.zeros(400, dtype=np.int8)
    one[:200] = 1
    p._frame_matrices = frames(np.random.default_rng(1).random(400), one)
    out = p.compute_frame_based_mAP()
    assert out["mAP"] is None
    assert out["undefined_AP"] == ["c0"]


def test_absent_class_still_scores_zero_not_undefined():
    """A class with no GT at all is a different case, and keeps its old 0.0."""
    p = evaluator()
    truth = np.zeros((300, 2), dtype=np.int8)
    truth[:50, 0] = 1                     # class b never occurs
    scores = np.random.default_rng(2).random((300, 2)).astype(np.float32)
    p._frame_matrices = (scores, truth, truth.any(axis=1), ["a", "b"])
    out = p.compute_frame_based_mAP()
    assert out["per_class_AP"]["b"] == 0.0
    assert out["undefined_AP"] == []
    assert out["mAP"] == pytest.approx(out["per_class_AP"]["a"])


def test_empty_split_reports_nothing_rather_than_zero():
    p = evaluator()
    p._frame_matrices = (np.zeros((0, 1), dtype=np.float32),
                         np.zeros((0, 1), dtype=np.int8),
                         np.zeros(0, dtype=bool), ["c0"])
    out = p.compute_frame_based_mAP()
    assert out["mAP"] is None


def test_shipped_configs_all_score_background_frames():
    """The omission was per-preset, so the setting now lives in the shared base."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    base = (root / "configs" / "_dataset.py").read_text()
    tree = next(node for node in ast.parse(base).body
                if isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", None) == "evaluation")
    keys = {k.arg: getattr(k.value, "value", None) for k in tree.value.keywords}
    assert keys.get("map_frame_filter") == "all"
    assert keys.get("ap_mode") == "sklearn"

    # No preset may quietly ask for the blind variant.
    for config in (root / "configs").glob("*.py"):
        text = config.read_text()
        assert "map_frame_filter=\"nonempty_gt\"" not in text, config.name
        assert "map_frame_filter='nonempty_gt'" not in text, config.name
