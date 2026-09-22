"""mAP has to be able to tell two models apart, because it picks the checkpoint.

`tools/train.py` keeps the epoch with the highest validation mAP. A metric that
is pinned to the same value whatever the model says turns that into a coin toss
— and worse than a coin toss, since the comparison is a strict `>`, so the
earliest scored epoch wins every tie and the least-trained model is published.

That is what the removed `map_frame_filter='nonempty_gt'` variant did on a
one-behaviour dataset: it kept only frames carrying a label, every one of those
is a positive for the only class, and AP over an all-positive column is 1.0 by
arithmetic. Every frame is scored now, background included.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from vtrace.evaluations.precision import Precision

ROOT = Path(__file__).resolve().parent.parent


def evaluator(ap_mode="sklearn"):
    """A Precision with the plumbing bypassed — these tests drive the maths directly."""
    p = Precision.__new__(Precision)
    p.ap_mode = ap_mode
    return p


def frames(scores, truth):
    """(scores, gt_binary, labels) as `_build_frame_matrices` returns it."""
    scores = np.asarray(scores, dtype=np.float32).reshape(len(scores), -1)
    truth = np.asarray(truth, dtype=np.int8).reshape(scores.shape)
    return scores, truth, [f"c{i}" for i in range(scores.shape[1])]


@pytest.mark.parametrize("scores, name", [
    (np.linspace(0, 1, 200), "ranked well"),
    (np.linspace(1, 0, 200), "ranked backwards"),
    (np.zeros(200), "no signal at all"),
])
def test_an_all_positive_column_has_no_measurable_ap(scores, name):
    """The three models below are not equally good; AP over positives cannot say so."""
    p = evaluator()
    ap = p.compute_average_precision(np.asarray(scores), np.ones(200, dtype=np.int8))
    assert ap == pytest.approx(1.0), f"{name}: AP is pinned, which is the premise"

    p._frame_matrices = frames(scores, np.ones(200))
    out = p.compute_frame_based_mAP()
    assert out["mAP"] is None, f"{name}: an unmeasurable split must not report a score"
    assert out["undefined_AP"] == ["c0"]
    assert out["per_class_AP"]["c0"] is None


def test_background_frames_make_a_single_class_measurable():
    """Same predictions, background frames scored: now the ranking matters."""
    rng = np.random.default_rng(0)
    truth = np.zeros(2000, dtype=np.int8)
    truth[:100] = 1                       # 5% positives, like a behaviour corpus

    good = np.where(truth == 1, 0.9, 0.1) + rng.normal(0, 0.01, 2000)
    middling = rng.random(2000)
    backwards = np.where(truth == 1, 0.1, 0.9) + rng.normal(0, 0.01, 2000)

    aps = []
    for scores in (good, middling, backwards):
        p = evaluator()
        p._frame_matrices = frames(scores, truth)
        out = p.compute_frame_based_mAP()
        assert out["mAP"] is not None and out["undefined_AP"] == []
        aps.append(out["mAP"])

    assert aps[0] > aps[1] > aps[2], aps
    assert aps[0] > 0.9 and aps[2] < 0.1


def test_confident_false_positives_in_the_background_cost_something():
    """The reason the background frames are scored at all.

    AP is a ranking measure, so a false positive only hurts when it outranks a
    true one. Scoring the background is what lets those false positives into the
    ranking in the first place — drop those frames and the two models below are
    indistinguishable, which is exactly the blindness the old variant had.
    """
    truth = np.zeros(1000, dtype=np.int8)
    truth[:50] = 1
    rng = np.random.default_rng(4)
    clean = np.where(truth == 1, rng.uniform(0.5, 0.9, 1000), 0.0).astype(np.float32)
    trigger_happy = clean.copy()
    trigger_happy[500:800] = 0.95         # confident nonsense, above every true positive

    scored = []
    for scores in (clean, trigger_happy):
        p = evaluator()
        p._frame_matrices = frames(scores, truth)
        scored.append(p.compute_frame_based_mAP()["mAP"])
    assert scored[0] == pytest.approx(1.0)
    assert scored[1] < 0.3, scored

    # The same two models, scored only on frames that carry a label: identical,
    # and both a perfect 1.0. Nothing to choose between them.
    blind = [evaluator().compute_average_precision(s[truth == 1], np.ones(50, dtype=np.int8))
             for s in (clean, trigger_happy)]
    assert blind == [pytest.approx(1.0), pytest.approx(1.0)]


def test_the_removed_variant_is_refused_with_a_reason():
    """And refused on the option itself, not on some unrelated path check."""
    with pytest.raises(ValueError, match="no longer supports"):
        Precision(ground_truth_filename="", prediction_filename={}, subset="validation",
                  tiou_thresholds=[0.5], map_frame_filter="nonempty_gt")


def test_a_config_still_naming_the_default_keeps_working():
    """Configs outside this checkout pass map_frame_filter='all'; that is a no-op now."""
    import inspect
    signature = inspect.signature(Precision.__init__)
    assert signature.parameters["map_frame_filter"].default is None
    # 'all' must not raise — construction fails later on the dummy paths, not here.
    with pytest.raises(Exception) as caught:
        Precision(ground_truth_filename="", prediction_filename={}, subset="validation",
                  tiou_thresholds=[0.5], map_frame_filter="all")
    assert "map_frame_filter" not in str(caught.value)


def test_an_absent_class_still_scores_zero_not_undefined():
    """A class with no GT at all is a different case, and keeps its old 0.0."""
    p = evaluator()
    truth = np.zeros((300, 2), dtype=np.int8)
    truth[:50, 0] = 1                     # class b never occurs
    scores = np.random.default_rng(2).random((300, 2)).astype(np.float32)
    p._frame_matrices = (scores, truth, ["a", "b"])
    out = p.compute_frame_based_mAP()
    assert out["per_class_AP"]["b"] == 0.0
    assert out["undefined_AP"] == []
    assert out["mAP"] == pytest.approx(out["per_class_AP"]["a"])


def test_an_empty_split_reports_nothing_rather_than_zero():
    p = evaluator()
    p._frame_matrices = (np.zeros((0, 1), dtype=np.float32),
                         np.zeros((0, 1), dtype=np.int8), ["c0"])
    assert p.compute_frame_based_mAP()["mAP"] is None


def test_no_shipped_config_names_the_removed_option():
    for config in (ROOT / "configs").glob("*.py"):
        assert "map_frame_filter" not in config.read_text(), config.name


def test_the_shared_config_pins_the_ap_convention():
    base = (ROOT / "configs" / "_dataset.py").read_text()
    tree = next(node for node in ast.parse(base).body
                if isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", None) == "evaluation")
    keys = {k.arg: getattr(k.value, "value", None) for k in tree.value.keywords}
    assert keys.get("ap_mode") == "sklearn"
