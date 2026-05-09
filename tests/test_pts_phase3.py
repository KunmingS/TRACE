"""Phase 3 tests for the inference path.

Covers ``tools/infer.py``:

1. ``probe_video`` builds and caches the PTS table and reports
   PTS-derived ``num_frames`` / ``duration`` / ``fps``.
2. A second call reuses the cache rather than rebuilding from the
   container.
3. When the container reports an invalid average fps (some weird MKV
   files), the helper falls back to estimating fps from the PTS span
   instead of dividing by zero / a negative.
4. ``generate_pseudo_annotations`` emits cached-clip entries while preserving
   ``source_video`` + ``source_frame_offset`` + ``source_pts_table`` so the
   existing Phase 2 PTS plumbing stays active.
5. Bad video paths get skipped with a warning rather than aborting the
   whole inference run.
6. The pseudo-annotation is consumable by Phase 2's
   ``Precision._load_clip_pts`` — the end-to-end PTS thread from
   ``tools/infer.py`` to ``trace_tad.evaluations.precision`` is intact.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
DEV_CFR_VIDEO = ROOT / "data" / "dev_test" / "clips" / "2025-06-27_14_51_47_clip_158.mp4"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

pytestmark = pytest.mark.skipif(
    not DEV_CFR_VIDEO.is_file(),
    reason=f"requires CFR fixture {DEV_CFR_VIDEO} (dev_test dataset)",
)


def _silent_logger():
    """A logger that drops everything — we don't want test output noise."""
    logger = logging.getLogger("test_pts_phase3")
    logger.handlers = [logging.NullHandler()]
    logger.setLevel(logging.CRITICAL)
    return logger


def _copy_video(src: Path, dst: Path) -> Path:
    shutil.copy(src, dst)
    return dst


def _stub_clip_materializer(monkeypatch):
    import trace_tad.data_prep as data_prep

    def fake_materialize(video_path, output_dir, *, clip_stem=None, **_kwargs):
        cache_dir = Path(output_dir) / "cache" / "videos"
        cache_dir.mkdir(parents=True, exist_ok=True)
        clip_path = cache_dir / f"{clip_stem}_clip_0.mp4"
        clip_path.write_bytes(b"cached clip")
        return [{
            "clip_idx": 0,
            "frame": 768,
            "duration": 768 / 30.0,
            "source_video": os.path.abspath(video_path),
            "source_frame_offset": 0,
            "source_start_seconds": 0.0,
            "source_pts_table": os.path.abspath(f"{video_path}.pts.npy"),
            "cached_video": os.fspath(clip_path),
        }]

    monkeypatch.setattr(data_prep, "materialize_video_clips", fake_materialize)


# ─────────────────────────────────────────────────────────────────────
# probe_video
# ─────────────────────────────────────────────────────────────────────


def test_probe_video_builds_and_caches_pts(tmp_path):
    from infer import probe_video

    vid = _copy_video(DEV_CFR_VIDEO, tmp_path / "cfr.mp4")
    pts_cache = tmp_path / "cfr.mp4.pts.npy"
    assert not pts_cache.exists()

    n, duration, fps, pts_path = probe_video(os.fspath(vid))

    assert n == 768
    assert fps == pytest.approx(30.0, abs=1e-3)
    # Duration on a 768-frame, 30fps file should be very close to 768/30
    # via the PTS-span + 1/fps formula.
    assert duration == pytest.approx(768 / 30.0, abs=1e-3)

    # Cache file must exist and the returned path must point to it.
    assert pts_cache.is_file()
    assert pts_path == os.fspath(pts_cache.resolve())


def test_probe_video_second_call_reuses_pts_cache(tmp_path):
    from infer import probe_video

    vid = _copy_video(DEV_CFR_VIDEO, tmp_path / "cfr.mp4")
    pts_cache = tmp_path / "cfr.mp4.pts.npy"

    probe_video(os.fspath(vid))
    assert pts_cache.is_file()
    cache_mtime = pts_cache.stat().st_mtime

    # Second call must NOT rewrite the cache.
    probe_video(os.fspath(vid))
    assert pts_cache.stat().st_mtime == cache_mtime


def test_probe_video_falls_back_when_avg_fps_invalid(tmp_path, monkeypatch):
    """Some containers report ``avg_fps == 0`` — in that case, derive
    fps from the PTS span instead of returning a degenerate value."""
    import decord
    from infer import probe_video

    vid = _copy_video(DEV_CFR_VIDEO, tmp_path / "cfr.mp4")

    # Class-level patch: every VideoReader instance probe_video constructs
    # reports 0 fps, exercising the PTS-span fallback in probe_video. The
    # PTS table itself still comes from the real container (via
    # data_prep._load_or_build_pts).
    monkeypatch.setattr(
        decord.VideoReader, "get_avg_fps", lambda self: 0.0, raising=False
    )

    n, duration, fps, pts_path = probe_video(os.fspath(vid))

    assert n == 768
    # Fallback fps = (n - 1) / pts_span. On a 30-fps clip pts_span is
    # 25.566…s, so fps ≈ 30.01.
    assert fps == pytest.approx(30.0, abs=0.1)
    assert duration > 0
    assert pts_path is not None and os.path.isfile(pts_path)


# ─────────────────────────────────────────────────────────────────────
# generate_pseudo_annotations
# ─────────────────────────────────────────────────────────────────────


def test_generate_pseudo_annotations_emits_cached_clip_schema(tmp_path, monkeypatch):
    from infer import generate_pseudo_annotations

    _stub_clip_materializer(monkeypatch)
    vid = _copy_video(DEV_CFR_VIDEO, tmp_path / "cfr_video.mp4")
    db = generate_pseudo_annotations(
        [os.fspath(vid)],
        _silent_logger(),
        cache_dir=os.fspath(tmp_path / "predict"),
    )

    assert "database" in db
    assert "cfr_video_clip_0" in db["database"]
    entry = db["database"]["cfr_video_clip_0"]

    # Standard fields.
    assert entry["subset"] == "validation"
    assert entry["frame"] == 768
    assert entry["duration"] == pytest.approx(768 / 30.0, abs=1e-3)
    assert entry["annotations"] == []

    # Cached-clip decode with source timeline metadata.
    assert entry["cached_video"].endswith("cfr_video_clip_0.mp4")
    assert entry["source_video"] == os.path.abspath(vid)
    assert entry["source_frame_offset"] == 0
    assert entry["source_prediction_name"] == "cfr_video"
    assert entry["source_start_seconds"] == 0.0
    assert "source_pts_table" in entry, (
        "source_pts_table must be in the pseudo-annotation entry so "
        "convert_to_seconds (Phase 2) takes the PTS path"
    )
    assert entry["source_pts_table"].endswith(".pts.npy")
    assert os.path.isfile(entry["source_pts_table"])


def test_generate_pseudo_annotations_skips_bad_videos(tmp_path, monkeypatch):
    from infer import generate_pseudo_annotations

    _stub_clip_materializer(monkeypatch)
    good = _copy_video(DEV_CFR_VIDEO, tmp_path / "good.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not actually a video file")

    db = generate_pseudo_annotations(
        [os.fspath(good), os.fspath(bad)], _silent_logger()
    )

    assert "good_clip_0" in db["database"]
    assert "bad" not in db["database"], "broken file must be skipped, not recorded"


def test_generate_pseudo_annotations_raises_when_all_videos_bad(tmp_path):
    from infer import generate_pseudo_annotations

    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not actually a video file")

    with pytest.raises(RuntimeError, match="No videos could be probed"):
        generate_pseudo_annotations([os.fspath(bad)], _silent_logger())


# ─────────────────────────────────────────────────────────────────────
# Phase 2 ↔ Phase 3 integration
# ─────────────────────────────────────────────────────────────────────


def test_pseudo_annotation_pts_table_is_loadable_by_precision(tmp_path, monkeypatch):
    """``Precision._load_clip_pts`` (Phase 2) must accept the entry
    shape that ``generate_pseudo_annotations`` (Phase 3) emits, so the
    PTS thread is unbroken end-to-end."""
    from infer import generate_pseudo_annotations
    from trace_tad.evaluations.precision import _load_clip_pts

    _stub_clip_materializer(monkeypatch)
    vid = _copy_video(DEV_CFR_VIDEO, tmp_path / "vid.mp4")
    db = generate_pseudo_annotations(
        [os.fspath(vid)],
        _silent_logger(),
        cache_dir=os.fspath(tmp_path / "predict"),
    )
    entry = db["database"]["vid_clip_0"]

    clip_pts = _load_clip_pts(entry)

    assert clip_pts is not None
    assert clip_pts.dtype == np.float64
    assert len(clip_pts) == entry["frame"]
    # First frame is rebased to 0.
    assert clip_pts[0] == 0.0
    # Last frame on a 768-frame, 30fps source = 767/30 ≈ 25.5667s.
    np.testing.assert_allclose(clip_pts[-1], 767 / 30.0, atol=1e-3)


def test_aggregate_clip_predictions_merges_source_timeline():
    from infer import aggregate_clip_predictions

    db = {
        "vid_clip_0": {
            "source_prediction_name": "vid",
            "source_start_seconds": 0.0,
        },
        "vid_clip_1": {
            "source_prediction_name": "vid",
            "source_start_seconds": 25.6,
        },
    }
    raw = {
        "vid_clip_1": [{"segment": [0.4, 1.0], "label": "walk", "score": 0.9}],
        "vid_clip_0": [{"segment": [2.0, 3.0], "label": "rest", "score": 0.8}],
    }

    merged = aggregate_clip_predictions(raw, db)

    assert list(merged) == ["vid"]
    assert merged["vid"] == [
        {"segment": [2.0, 3.0], "label": "rest", "score": 0.8},
        {"segment": [26.0, 26.6], "label": "walk", "score": 0.9},
    ]
