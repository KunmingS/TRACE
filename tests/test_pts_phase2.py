"""Phase 2 tests for PTS-based timestamp ↔ frame mapping in the
training/eval/post-processing path.

Covers:

1. ``post_processing.utils.convert_to_seconds`` —
   - PTS-aware path produces clip-relative seconds via piecewise-linear
     interpolation against the per-clip PTS array.
   - On a synthetic CFR PTS table the PTS path is bit-equivalent
     (within float precision) to the legacy ``fps`` path.
   - On a synthesized non-uniform (VFR) PTS table the PTS path
     diverges from the CFR formula in the direction predicted by hand.
   - Legacy fallback (no ``source_pts_table`` in meta) preserves the
     historic behaviour exactly.
2. ``evaluations.precision.Precision._process_single_video_prediction``
   PTS-aware → predictions in seconds get binned through
   ``searchsorted(clip_pts, t)`` instead of ``t * eval_fps``. On a
   synthetic VFR clip the resulting frame bins differ from the CFR
   approximation.
3. ``evaluations.precision._load_clip_pts`` helper handles missing /
   broken ``source_pts_table`` gracefully (returns ``None``).
4. ``ThumosSlidingDataset.__getitem__`` propagates
   ``source_pts_table`` from ``video_info`` into the sample dict so
   the pipeline's ``Collect`` node passes it through to ``meta``.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────
# convert_to_seconds
# ─────────────────────────────────────────────────────────────────────


def _write_pts_npy(tmp_path: Path, pts: np.ndarray, name: str = "src.pts.npy") -> str:
    p = tmp_path / name
    np.save(p, pts.astype(np.float64))
    return os.fspath(p)


def _make_meta(*, pts_path=None, source_frame_offset=0, clip_frame_count=None,
               fps=30.0, snippet_stride=1, offset_frames=0,
               window_start_frame=0, duration=10.0):
    meta = dict(
        fps=fps,
        snippet_stride=snippet_stride,
        offset_frames=offset_frames,
        window_start_frame=window_start_frame,
        duration=duration,
    )
    if pts_path is not None:
        meta["source_pts_table"] = pts_path
        meta["source_frame_offset"] = source_frame_offset
        meta["clip_frame_count"] = (
            clip_frame_count
            if clip_frame_count is not None
            else len(np.load(pts_path)) - source_frame_offset
        )
    return meta


def test_convert_to_seconds_pts_matches_cfr_on_uniform_pts(tmp_path):
    """On a synthetic CFR PTS array, the PTS path and the legacy
    ``fps``-based path must produce the same seconds (within float
    rounding)."""
    from trace_tad.models.utils.post_processing.utils import convert_to_seconds

    fps = 30.0
    n = 256
    pts = np.arange(n, dtype=np.float64) / fps  # exact CFR
    pts_path = _write_pts_npy(tmp_path, pts)

    # Sub-frame model output indices, including some endpoints.
    seg = torch.tensor(
        [[0.0, 30.0], [15.5, 100.25], [200.0, 255.0]],
        dtype=torch.float32,
    )

    meta_pts = _make_meta(pts_path=pts_path, clip_frame_count=n,
                          fps=fps, duration=(n - 1) / fps)
    meta_legacy = _make_meta(fps=fps, duration=(n - 1) / fps)

    out_pts = convert_to_seconds(seg.clone(), meta_pts).numpy()
    out_legacy = convert_to_seconds(seg.clone(), meta_legacy).numpy()

    np.testing.assert_allclose(out_pts, out_legacy, atol=1e-6)


def test_convert_to_seconds_pts_diverges_from_cfr_on_vfr(tmp_path):
    """On a synthesized VFR PTS array, the PTS path lands on the
    container's actual presentation time while the CFR path computes a
    fictitious average-fps time. The two must disagree noticeably at a
    frame index that sits inside the gap."""
    from trace_tad.models.utils.post_processing.utils import convert_to_seconds

    # 10 frames: 5 frames at 20fps, then a 0.3s gap, then 5 more at 20fps.
    pts = np.array(
        [0.00, 0.05, 0.10, 0.15, 0.20,
         0.50, 0.55, 0.60, 0.65, 0.70],
        dtype=np.float64,
    )
    pts_path = _write_pts_npy(tmp_path, pts)
    n = len(pts)
    avg_fps = (n - 1) / (pts[-1] - pts[0])  # ≈ 12.857

    # Model frame index 5 = first frame after the gap. PTS says 0.50 s,
    # CFR-with-avg-fps says 5 / 12.857 ≈ 0.389 s.
    seg = torch.tensor([[5.0, 5.0]], dtype=torch.float32)

    meta_pts = _make_meta(pts_path=pts_path, clip_frame_count=n,
                          fps=avg_fps, duration=pts[-1])
    meta_legacy = _make_meta(fps=avg_fps, duration=pts[-1])

    out_pts = convert_to_seconds(seg.clone(), meta_pts).numpy().ravel()
    out_legacy = convert_to_seconds(seg.clone(), meta_legacy).numpy().ravel()

    np.testing.assert_allclose(out_pts[0], 0.50, atol=1e-6)
    # Legacy avg-fps path lands somewhere in [0.35, 0.45] — definitely
    # not at the actual PTS of frame 5.
    assert abs(out_legacy[0] - 0.50) > 0.05, (
        f"legacy CFR path should disagree on VFR; got {out_legacy[0]} vs 0.50"
    )
    assert abs(out_pts[0] - out_legacy[0]) > 0.05


def test_convert_to_seconds_legacy_path_unchanged_when_pts_absent():
    """When `source_pts_table` is not in meta, the function must produce
    exactly the historic `(idx * stride + start + offset) / fps` value."""
    from trace_tad.models.utils.post_processing.utils import convert_to_seconds

    seg = torch.tensor([[7.0, 13.0]], dtype=torch.float32)
    meta = _make_meta(fps=24.0, snippet_stride=4, offset_frames=8,
                      window_start_frame=16, duration=999.0)

    expected = (7.0 * 4 + 16 + 8) / 24.0  # = 52/24
    expected_end = (13.0 * 4 + 16 + 8) / 24.0
    out = convert_to_seconds(seg.clone(), meta).numpy().ravel()
    np.testing.assert_allclose(out[0], expected, atol=1e-6)
    np.testing.assert_allclose(out[1], expected_end, atol=1e-6)


def test_convert_to_seconds_uses_source_frame_offset(tmp_path):
    """The clip-relative PTS must rebase to zero at the clip's first
    frame, not at the source video's first frame."""
    from trace_tad.models.utils.post_processing.utils import convert_to_seconds

    # Source video has 100 frames at 30 fps; clip starts at frame 30.
    n_src = 100
    fps = 30.0
    pts = np.arange(n_src, dtype=np.float64) / fps
    pts_path = _write_pts_npy(tmp_path, pts)

    seg = torch.tensor([[0.0, 30.0]], dtype=torch.float32)
    meta = _make_meta(pts_path=pts_path, source_frame_offset=30,
                      clip_frame_count=60, fps=fps, duration=2.0)

    out = convert_to_seconds(seg.clone(), meta).numpy().ravel()
    # Clip-local frame 0 → 0.0 s, clip-local frame 30 → 1.0 s. The source
    # frame offset of 30 must NOT appear in the output.
    np.testing.assert_allclose(out[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[1], 1.0, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────
# Precision (eval) PTS-aware prediction binning
# ─────────────────────────────────────────────────────────────────────


def _build_gt_dataset(tmp_path: Path, *, pts: np.ndarray, frame_segment, label="bx"):
    """Lay out a minimal dataset.json + predictions.json for `Precision`."""
    pts_path = _write_pts_npy(tmp_path, pts, name="vfr.pts.npy")
    gt = {
        "database": {
            "clipA": {
                "subset": "validation",
                "duration": float(pts[-1] - pts[0]),
                "frame": int(len(pts)),
                "annotations": [{
                    "label": label,
                    "frame_segment": list(frame_segment),
                    "segment": [float(pts[frame_segment[0]] - pts[0]),
                                float(pts[frame_segment[1]] - pts[0])],
                    "timestamp_sec": [float(pts[frame_segment[0]] - pts[0]),
                                      float(pts[frame_segment[1]] - pts[0])],
                }],
                "source_video": "/dummy/vfr.mp4",
                "source_frame_offset": 0,
                "source_pts_table": pts_path,
            }
        }
    }
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    return gt_path, pts_path


def test_precision_uses_pts_for_prediction_binning(tmp_path):
    """A prediction in clip-relative seconds is binned through the
    clip's PTS array, not via ``eval_fps`` — so on a VFR clip the
    prediction frames match the GT frames exactly even when the
    average fps would mis-bin them."""
    from trace_tad.evaluations.precision import Precision

    # Synthesized VFR: 10 frames as in convert_to_seconds test.
    pts = np.array(
        [0.00, 0.05, 0.10, 0.15, 0.20,
         0.50, 0.55, 0.60, 0.65, 0.70],
        dtype=np.float64,
    )
    # GT: label "bx" on frames [5, 7] (= 0.50–0.60 s in clip-local PTS).
    gt_path, _ = _build_gt_dataset(tmp_path, pts=pts, frame_segment=[5, 7])

    # Prediction: same seconds as GT — clip-local 0.50–0.60 s. The
    # `prediction_filename` arg is misleadingly named; production passes
    # the dict directly (see cores/eval_engine.py:324).
    predictions = {
        "results": {
            "clipA": [
                {"segment": [0.50, 0.60], "label": "bx", "score": 0.9},
            ]
        }
    }

    ev = Precision(
        ground_truth_filename=os.fspath(gt_path),
        prediction_filename=predictions,
        subset="validation",
        tiou_thresholds=[0.5],
        eval_fps=12.857,  # the avg fps of this synthetic VFR — would mis-bin
    )

    # PTS table was loaded for clipA.
    assert ev.clip_pts["clipA"] is not None
    np.testing.assert_allclose(ev.clip_pts["clipA"], pts - pts[0])

    # The prediction must have populated frames 5–6 inclusive (searchsorted
    # right for 0.60 → 8, then end_frame=8 means slice [5:8] on rows 5,6,7).
    # Since GT marks frames 5–6 (frame_segment=[5,7] → range(5, 7)), we
    # check that the prediction's frame coverage matches the GT precisely.
    pred_label_sets = ev.pred_data["clipA"]
    assert pred_label_sets[5] == {"bx"}
    assert pred_label_sets[6] == {"bx"}
    # Frame 4 PTS = 0.20 — strictly less than 0.50, must NOT be tagged.
    assert pred_label_sets[4] == set()
    # Frame 7 PTS = 0.60 — searchsorted(right) on 0.60 = 8, so frames
    # [5,6,7] all get the label. (This matches the GT's frame 7 inclusion
    # via half-open semantics on the prediction side.)
    assert pred_label_sets[7] == {"bx"}


def test_precision_falls_back_to_eval_fps_for_legacy_dataset(tmp_path):
    """If the dataset has no ``source_pts_table``, the historic
    ``t * eval_fps`` rounding is preserved."""
    from trace_tad.evaluations.precision import Precision

    gt = {
        "database": {
            "clipL": {
                "subset": "validation",
                "duration": 10.0,
                "frame": 300,
                "annotations": [{
                    "label": "bx",
                    "frame_segment": [60, 90],
                    "segment": [2.0, 3.0],
                }],
                # No source_pts_table → legacy CFR path
            }
        }
    }
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")

    predictions = {
        "results": {
            "clipL": [{"segment": [2.0, 3.0], "label": "bx", "score": 0.8}]
        }
    }

    ev = Precision(
        ground_truth_filename=os.fspath(gt_path),
        prediction_filename=predictions,
        subset="validation",
        tiou_thresholds=[0.5],
        eval_fps=30.0,
    )

    assert ev.clip_pts["clipL"] is None  # fell back

    # 2.0 * 30 = 60, 3.0 * 30 = 90 — frames [60, 90) labeled "bx".
    pred_label_sets = ev.pred_data["clipL"]
    assert pred_label_sets[60] == {"bx"}
    assert pred_label_sets[89] == {"bx"}
    assert pred_label_sets[59] == set()
    assert pred_label_sets[90] == set()


def test_load_clip_pts_handles_missing_or_broken_pts_path(tmp_path):
    """`_load_clip_pts` must not raise on a missing or unreadable
    ``source_pts_table`` — return ``None`` so the caller falls back."""
    from trace_tad.evaluations.precision import _load_clip_pts

    # Missing file path.
    assert _load_clip_pts({"source_pts_table": "/no/such/file.npy",
                           "source_frame_offset": 0, "frame": 100}) is None

    # No source_pts_table at all (legacy entry).
    assert _load_clip_pts({"frame": 100}) is None

    # Broken .npy file (truncated header).
    bad = tmp_path / "broken.pts.npy"
    bad.write_bytes(b"\x00\x01not-a-real-numpy-file")
    assert _load_clip_pts({"source_pts_table": os.fspath(bad),
                           "source_frame_offset": 0, "frame": 100}) is None


# ─────────────────────────────────────────────────────────────────────
# thumos.py sample injection
# ─────────────────────────────────────────────────────────────────────


def test_thumos_sliding_dataset_injects_pts_table_into_sample(tmp_path, monkeypatch):
    """``ThumosSlidingDataset.__getitem__`` must forward
    ``source_pts_table`` from ``video_info`` into the sample so the
    pipeline's ``Collect`` step propagates it to ``meta``."""
    from trace_tad.datasets.thumos import ThumosSlidingDataset

    captured = {}

    def fake_pipeline(sample):
        captured.update(sample)
        return sample

    # Stub-construct the dataset without going through __init__ — we only
    # exercise __getitem__'s sample-building, not file I/O.
    ds = ThumosSlidingDataset.__new__(ThumosSlidingDataset)
    ds.pipeline = fake_pipeline
    ds.data_path = "/tmp/whatever"
    ds.snippet_stride = 4
    ds.offset_frames = 0
    ds.sample_stride = 1
    ds.window_size = 256
    pts_path = _write_pts_npy(tmp_path, np.arange(128) / 30.0)
    video_info = {
        "frame": 64,
        "duration": 64 / 30.0,
        "annotations": [],
        "source_video": "/abs/source.mp4",
        "source_frame_offset": 32,
        "source_pts_table": pts_path,
    }
    ds.data_list = [(
        "clipX",
        video_info,
        {},  # video_anno
        np.array([0, 4, 8, 12], dtype=np.int64),  # window_snippet_centers
    )]

    ds[0]

    assert captured["source_pts_table"] == pts_path
    assert captured["source_frame_offset"] == 32
    assert captured["clip_frame_count"] == 64
    # Legacy fps key still present for backward compat.
    assert captured["fps"] == pytest.approx(30.0)


def test_thumos_sliding_dataset_omits_pts_for_legacy_video_info(tmp_path):
    """A legacy ``video_info`` (no ``source_video``) must NOT inject
    PTS-related keys into the sample."""
    from trace_tad.datasets.thumos import ThumosSlidingDataset

    captured = {}

    def fake_pipeline(sample):
        captured.update(sample)
        return sample

    ds = ThumosSlidingDataset.__new__(ThumosSlidingDataset)
    ds.pipeline = fake_pipeline
    ds.data_path = "/tmp/whatever"
    ds.snippet_stride = 4
    ds.offset_frames = 0
    ds.sample_stride = 1
    ds.window_size = 256
    video_info = {
        "frame": 768,
        "duration": 25.6,
        "annotations": [],
        # No source_video → fully legacy entry.
    }
    ds.data_list = [("clipL", video_info, {},
                     np.array([0, 4, 8, 12], dtype=np.int64))]
    ds[0]

    assert "source_pts_table" not in captured
    assert "source_frame_offset" not in captured
    assert "clip_frame_count" not in captured
