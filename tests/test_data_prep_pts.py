"""Tests for the PTS-based timestamp ↔ frame mapping in ``data_prep``.

Phase 1 of the refactor described in
``docs/pts-based-frame-mapping.md``. Verifies that:

1. ``_load_or_build_pts`` correctly builds + caches a PTS table and
   invalidates the cache when the source mtime advances.
2. On a real CFR clip, PTS-based ``searchsorted`` agrees with the
   legacy ``round(t * fps)`` mapping to within 1 frame (sub-frame
   rounding only).
3. ``_process_video`` emits the new ``timestamp_sec`` /
   ``source_pts_table`` schema fields when virtual clips are enabled.
4. With a synthesized non-uniform PTS array (simulating a VFR
   recording), the PTS-derived ``frame_segment`` differs from what a
   CFR-assumption would produce — in the direction predicted by manual
   inspection.
5. With multiple clips and an annotation that crosses a clip boundary,
   each clip carries the correct slice (PTS-correct, properly clamped),
   ``source_frame_offset`` advances by ``clip_frames`` per clip, and
   ``duration`` is PTS-derived per clip.
6. ``prepare_dataset`` actually writes the new schema fields
   (``source_pts_table``, ``timestamp_sec``) into ``dataset.json`` on
   disk — not just in the in-memory return of ``_process_video``.
7. CSV annotations whose ``endTimestamp`` exceeds the video duration
   are clamped to the last available frame rather than overflowing.
"""

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
DEV_CFR_VIDEO = ROOT / "data" / "dev_test" / "clips" / "2025-06-27_14_51_47_clip_158.mp4"

pytestmark = pytest.mark.skipif(
    not DEV_CFR_VIDEO.is_file(),
    reason=f"requires CFR fixture {DEV_CFR_VIDEO} (dev_test dataset)",
)


def _copy_video(src: Path, dst: Path) -> Path:
    """Copy a video into a tmp_path so cache files don't pollute the repo."""
    import shutil
    shutil.copy(src, dst)
    return dst


def test_load_or_build_pts_caches_and_invalidates(tmp_path):
    from trace_tad.data_prep import _load_or_build_pts, _pts_cache_path

    vid = _copy_video(DEV_CFR_VIDEO, tmp_path / "cfr.mp4")
    cache = Path(_pts_cache_path(vid))
    assert not cache.exists()

    pts1 = _load_or_build_pts(vid)
    assert pts1.dtype == np.float64
    assert pts1.ndim == 1
    assert len(pts1) > 0
    assert cache.is_file()

    # Second call should hit the cache and return the identical array.
    pts2 = _load_or_build_pts(vid)
    np.testing.assert_array_equal(pts1, pts2)

    # Touch the source so its mtime is strictly newer than the cache.
    time.sleep(0.05)
    os.utime(vid, None)
    assert os.path.getmtime(vid) > os.path.getmtime(cache)

    pts3 = _load_or_build_pts(vid)
    np.testing.assert_array_equal(pts1, pts3)
    # After rebuild, cache mtime should now be ≥ source mtime again.
    assert os.path.getmtime(cache) >= os.path.getmtime(vid)


def test_pts_searchsorted_matches_cfr_fps_within_one_frame(tmp_path):
    """On a CFR file the PTS-based path must agree with `round(t × fps)`."""
    import decord
    from trace_tad.data_prep import _load_or_build_pts

    vid = _copy_video(DEV_CFR_VIDEO, tmp_path / "cfr.mp4")
    pts = _load_or_build_pts(vid)
    vr = decord.VideoReader(os.fspath(vid))
    fps = vr.get_avg_fps()
    n = len(vr)

    rng = np.random.default_rng(0)
    ts = rng.uniform(0.0, n / fps, size=200)
    idx_cfr = np.round(ts * fps).astype(int).clip(0, n - 1)
    idx_pts = np.searchsorted(pts, ts).clip(0, n - 1)

    # Sub-frame rounding only — never more than 1 frame off on a CFR file.
    assert int(np.max(np.abs(idx_cfr - idx_pts))) <= 1


def test_process_video_emits_new_schema_fields_for_virtual_clip(tmp_path):
    """End-to-end: real CFR clip + synthetic CSV → schema includes
    ``timestamp_sec`` and ``source_pts_table``.
    """
    from trace_tad.data_prep import _process_video

    vid = _copy_video(DEV_CFR_VIDEO, tmp_path / "cfr.mp4")
    csv_path = tmp_path / "cfr.csv"
    # The dev_test clip is ~25.6s @ 30fps. Pick a short window comfortably
    # inside it so the annotation falls within clip 0 of the new prep.
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["labelId", "timestamp", "endTimestamp"])
        w.writerow(["drinking", "1.0", "2.0"])

    clips_data = _process_video(
        os.fspath(vid),
        os.fspath(csv_path),
        os.fspath(tmp_path / "clips_out"),
        clip_frames=768,
        virtual_clips=True,
    )

    assert 0 in clips_data, "expected at least one clip with the seeded annotation"
    entry = clips_data[0]

    # Virtual-clip metadata is present.
    assert entry["source_video"] == os.path.abspath(vid)
    assert entry["source_frame_offset"] == 0
    assert "source_pts_table" in entry
    assert entry["source_pts_table"].endswith(".pts.npy")
    assert os.path.isfile(entry["source_pts_table"])

    # Annotation carries both legacy `segment` and new `timestamp_sec`.
    assert len(entry["annotations"]) == 1
    anno = entry["annotations"][0]
    assert anno["label"] == "drinking"
    assert "frame_segment" in anno
    assert "segment" in anno
    assert "timestamp_sec" in anno
    # `timestamp_sec` is the canonical PTS-derived value; `segment` is its
    # backward-compatible alias.
    assert anno["timestamp_sec"] == anno["segment"]

    seg_lo, seg_hi = anno["timestamp_sec"]
    # PTS-snapped value should land within ±2 frames (~67 ms) of the CSV
    # request (round-down on left, round-up on right boundary).
    assert abs(seg_lo - 1.0) < 0.1
    assert abs(seg_hi - 2.0) < 0.1


def test_process_video_uses_pts_for_vfr_mapping(tmp_path, monkeypatch):
    """With a synthesized non-uniform PTS table, the PTS-derived frame
    indices must reflect the actual frame timing — *not* the average fps.
    """
    from trace_tad import data_prep

    vid = _copy_video(DEV_CFR_VIDEO, tmp_path / "vfr_synth.mp4")
    csv_path = tmp_path / "vfr_synth.csv"

    # Synthetic VFR PTS: frames 0–4 at 20fps, then a 0.3s gap, then frames
    # 5–9 at 20fps. avg_fps = (10 - 1) / 0.7 ≈ 12.86, but no frame is
    # actually presented in [0.2 s, 0.5 s).
    fake_pts = np.array(
        [0.00, 0.05, 0.10, 0.15, 0.20,
         0.50, 0.55, 0.60, 0.65, 0.70],
        dtype=np.float64,
    )
    monkeypatch.setattr(
        data_prep, "_load_or_build_pts", lambda _path: fake_pts.copy()
    )
    # And also intercept the PTS cache path so the test stays in tmp_path.
    monkeypatch.setattr(
        data_prep,
        "_pts_cache_path",
        lambda _path: os.fspath(tmp_path / "vfr_synth.pts.npy"),
    )

    # Annotation at t=0.4s (= inside the gap). Under naive
    # `round(t × avg_fps)`, this would map to frame round(0.4 × 12.86) ≈ 5;
    # under PTS searchsorted it must map to frame 5 too — but for the *end*
    # at t=0.55 PTS gives frame 6 while `round(0.55 × 12.86) = 7`.
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["labelId", "timestamp", "endTimestamp"])
        w.writerow(["bx", "0.40", "0.55"])

    clips_data = data_prep._process_video(
        os.fspath(vid),
        os.fspath(csv_path),
        os.fspath(tmp_path / "clips_out"),
        clip_frames=768,  # > 10, so single clip
        virtual_clips=True,
    )

    assert 0 in clips_data
    anno = clips_data[0]["annotations"][0]
    assert anno["label"] == "bx"
    rel_start, rel_end = anno["frame_segment"]

    # PTS-correct mapping for our synthesized VFR:
    #   searchsorted(pts, 0.40, "left")          = 5
    #   searchsorted(pts, 0.55, "right") - 1     = 6
    assert rel_start == 5, f"expected frame 5, got {rel_start}"
    assert rel_end == 6, f"expected frame 6, got {rel_end}"

    # And the seconds reported are the actual PTS deltas, not avg-fps math.
    seg_lo, seg_hi = anno["timestamp_sec"]
    np.testing.assert_allclose(seg_lo, fake_pts[5] - fake_pts[0])
    np.testing.assert_allclose(seg_hi, fake_pts[6] - fake_pts[0])


def test_process_video_splits_annotations_across_clip_boundaries(tmp_path):
    """A 768-frame CFR video sliced into 3 clips of 300 frames each.

    Two annotations:

    - ``drink``: 8.0–12.0 s → source frames 240–360. Spans the
      300-frame boundary, so it must show up in **clip 0** (clamped to
      rel_end = 299) **and** **clip 1** (rel_start = 0).
    - ``eat``: 22.0–24.0 s → source frames 660–720. Lives entirely in
      **clip 2**, with offset bookkeeping on ``source_frame_offset``.

    This is the scenario where a regression in either ``clip_start_pts``
    indexing or the ``rel_start = max(0, ...)`` / ``rel_end = min(...)``
    clamps would silently corrupt training data — a single-clip test
    can't catch it.
    """
    from trace_tad.data_prep import _process_video

    vid = _copy_video(DEV_CFR_VIDEO, tmp_path / "cfr_multiclip.mp4")
    csv_path = tmp_path / "cfr_multiclip.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["labelId", "timestamp", "endTimestamp"])
        w.writerow(["drink", "8.0", "12.0"])  # spans frames 240..360
        w.writerow(["eat", "22.0", "24.0"])   # frames 660..720

    clip_frames = 300
    clips_data = _process_video(
        os.fspath(vid),
        os.fspath(csv_path),
        os.fspath(tmp_path / "clips_out"),
        clip_frames=clip_frames,
        virtual_clips=True,
    )

    # All three clips should have at least one annotation.
    assert set(clips_data.keys()) == {0, 1, 2}, (
        f"expected clips {{0, 1, 2}}, got {sorted(clips_data.keys())}"
    )

    # ── Clip 0: tail half of "drink", clamped at clip end ───────────
    c0 = clips_data[0]
    assert c0["frame"] == 300
    assert c0["source_frame_offset"] == 0
    drink0 = next(a for a in c0["annotations"] if a["label"] == "drink")
    assert drink0["frame_segment"] == [240, 299], drink0["frame_segment"]
    # PTS-derived seconds are clip-relative.
    np.testing.assert_allclose(drink0["timestamp_sec"][0], 8.0, atol=1e-3)
    np.testing.assert_allclose(drink0["timestamp_sec"][1], 299 / 30.0, atol=1e-3)
    assert drink0["timestamp_sec"] == drink0["segment"]

    # ── Clip 1: head half of "drink", starting at rel 0 ─────────────
    c1 = clips_data[1]
    assert c1["frame"] == 300
    assert c1["source_frame_offset"] == clip_frames  # = 300
    drink1 = next(a for a in c1["annotations"] if a["label"] == "drink")
    assert drink1["frame_segment"] == [0, 60], drink1["frame_segment"]
    # Clip-relative seconds: clip 1 starts at pts[300] = 10.0 s.
    np.testing.assert_allclose(drink1["timestamp_sec"][0], 0.0, atol=1e-3)
    np.testing.assert_allclose(drink1["timestamp_sec"][1], 12.0 - 10.0, atol=1e-3)

    # ── Clip 2: only "eat", and clip is shorter (tail clip: 168 frames) ─
    c2 = clips_data[2]
    assert c2["frame"] == 768 - 600  # 168
    assert c2["source_frame_offset"] == 2 * clip_frames  # = 600
    eat2 = next(a for a in c2["annotations"] if a["label"] == "eat")
    assert eat2["frame_segment"] == [60, 120], eat2["frame_segment"]
    np.testing.assert_allclose(eat2["timestamp_sec"][0], 22.0 - 20.0, atol=1e-3)
    np.testing.assert_allclose(eat2["timestamp_sec"][1], 24.0 - 20.0, atol=1e-3)

    # ── Per-clip duration must be PTS-derived, not assumed CFR @ 30 ─
    # For a CFR-30 source, full clip → 300/30 = 10.0 s, tail clip → 168/30 = 5.6 s.
    np.testing.assert_allclose(c0["duration"], 300 / 30.0, atol=1e-3)
    np.testing.assert_allclose(c1["duration"], 300 / 30.0, atol=1e-3)
    np.testing.assert_allclose(c2["duration"], 168 / 30.0, atol=1e-3)

    # No clip should leak the OTHER label.
    assert all(a["label"] == "drink" for a in c0["annotations"])
    assert all(a["label"] == "drink" for a in c1["annotations"])
    assert all(a["label"] == "eat" for a in c2["annotations"])


def test_prepare_dataset_writes_pts_fields_to_dataset_json(tmp_path):
    """End-to-end: ``prepare_dataset`` should persist the new schema
    (``source_pts_table``, ``timestamp_sec``) into ``dataset.json`` on
    disk. Phase 2/3 loaders read these fields off the JSON file, not
    from the in-memory return of ``_process_video``.
    """
    from trace_tad.data_prep import prepare_dataset

    # Lay out a self-contained mini-dataset in tmp_path: video + CSV.
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    src_vid = dataset_dir / "vid.mp4"
    src_csv = dataset_dir / "vid.csv"
    _copy_video(DEV_CFR_VIDEO, src_vid)
    with src_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["labelId", "timestamp", "endTimestamp"])
        w.writerow(["drink", "1.0", "2.0"])

    model_dir, json_path, classmap_path = prepare_dataset(
        os.fspath(dataset_dir),
        clip_frames=768,
        train_ratio=0.8,
    )

    assert Path(model_dir).name.startswith("model_")
    assert os.path.isfile(json_path)
    assert os.path.isfile(classmap_path)

    with open(json_path, "r", encoding="utf-8") as f:
        database = json.load(f)["database"]

    assert len(database) == 1
    (clip_key, entry), = database.items()
    assert clip_key.startswith("vid_clip_")

    # Virtual-clip metadata is on disk.
    assert entry["source_video"] == os.path.abspath(src_vid)
    assert entry["source_frame_offset"] == 0
    assert "source_pts_table" in entry, (
        "source_pts_table must be persisted into dataset.json — "
        "downstream Phase 2 loaders depend on it"
    )
    assert entry["source_pts_table"].endswith(".pts.npy")
    assert os.path.isfile(entry["source_pts_table"])

    # New per-annotation field is on disk too.
    anno = entry["annotations"][0]
    assert anno["label"] == "drink"
    assert "timestamp_sec" in anno, "timestamp_sec must be persisted"
    assert anno["timestamp_sec"] == anno["segment"]

    # Classmap was generated.
    classes = Path(classmap_path).read_text(encoding="utf-8").split()
    assert "drink" in classes


def test_process_video_clamps_csv_overflow_to_last_frame(tmp_path):
    """A CSV ``endTimestamp`` past the end of the video must clamp to
    the last available frame, not overflow into negative / out-of-range
    indices.
    """
    from trace_tad.data_prep import _process_video

    vid = _copy_video(DEV_CFR_VIDEO, tmp_path / "cfr_overflow.mp4")
    csv_path = tmp_path / "cfr_overflow.csv"
    # The dev_test clip is exactly 768 frames @ ~30 fps (≈ 25.6 s).
    # endTimestamp = 999.0 is well past the end of the file.
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["labelId", "timestamp", "endTimestamp"])
        w.writerow(["drink", "5.0", "999.0"])

    clips_data = _process_video(
        os.fspath(vid),
        os.fspath(csv_path),
        os.fspath(tmp_path / "clips_out"),
        clip_frames=768,  # single clip → tail equals full clip
        virtual_clips=True,
    )

    assert 0 in clips_data
    entry = clips_data[0]
    anno = entry["annotations"][0]
    rel_start, rel_end = anno["frame_segment"]

    # Lower bound: t=5.0s → frame 150 on the CFR fixture.
    assert rel_start == 150, f"expected frame 150, got {rel_start}"
    # Upper bound clamps to the last frame in the clip (= total_frames - 1
    # for this single-clip case = 767).
    assert rel_end == 767, (
        f"expected clamp to 767 (last frame), got {rel_end} — overflow not handled"
    )

    # PTS-derived seconds match the clamped frame, not 999 s.
    seg_hi = anno["timestamp_sec"][1]
    assert seg_hi < 30.0, f"timestamp_sec must be clamped, got {seg_hi}"
    np.testing.assert_allclose(seg_hi, 767 / 30.0, atol=1e-3)
