import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _touch_pair(folder: Path, stem: str, video_ext: str = ".mp4"):
    (folder / f"{stem}{video_ext}").write_bytes(b"")
    (folder / f"{stem}.csv").write_text("labelId\n", encoding="utf-8")


def _write_tiny_video(path: Path, frame_count: int = 6, fps: float = 3.0):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (16, 16),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot create a tiny mp4 fixture in this environment")
    try:
        for idx in range(frame_count):
            frame = np.full((16, 16, 3), idx * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_materialize_video_clips_writes_every_window(tmp_path, monkeypatch):
    pytest.importorskip("decord")
    from trace_tad import data_prep

    video = tmp_path / "source.mp4"
    _write_tiny_video(video, frame_count=5, fps=1.0)

    def fake_extract(_video_path, clip_path, *_args, **_kwargs):
        Path(clip_path).parent.mkdir(parents=True, exist_ok=True)
        Path(clip_path).write_bytes(b"clip")
        return True, None

    monkeypatch.setattr(data_prep, "_extract_cached_video_clip", fake_extract)

    clips = data_prep.materialize_video_clips(
        os.fspath(video),
        os.fspath(tmp_path / "predict"),
        clip_frames=2,
        cache_resolution=16,
        clip_stem="source",
    )

    assert [clip["frame"] for clip in clips] == [2, 2, 1]
    assert [clip["source_frame_offset"] for clip in clips] == [0, 2, 4]
    assert [round(clip["source_start_seconds"], 3) for clip in clips] == [0.0, 2.0, 4.0]
    cache_dir = tmp_path / "source.mp4.trace-clips" / "f2_r16_crf23"
    assert all(Path(clip["cached_video"]).is_file() for clip in clips)
    assert all(Path(clip["cached_video"]).parent == cache_dir for clip in clips)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["clip_frames"] == 2
    assert manifest["cache_resolution"] == 16


def test_materialize_video_clips_reuses_sidecar_cache(tmp_path, monkeypatch):
    pytest.importorskip("decord")
    from trace_tad import data_prep

    video = tmp_path / "source.mp4"
    _write_tiny_video(video, frame_count=5, fps=1.0)
    extract_calls = []

    def fake_extract(_video_path, clip_path, *_args, **_kwargs):
        extract_calls.append(Path(clip_path).name)
        Path(clip_path).parent.mkdir(parents=True, exist_ok=True)
        Path(clip_path).write_bytes(b"clip")
        return True, None

    monkeypatch.setattr(data_prep, "_extract_cached_video_clip", fake_extract)

    first = data_prep.materialize_video_clips(
        os.fspath(video),
        os.fspath(tmp_path / "predict_a"),
        clip_frames=2,
        cache_resolution=16,
        clip_stem="source",
    )
    second = data_prep.materialize_video_clips(
        os.fspath(video),
        os.fspath(tmp_path / "predict_b"),
        clip_frames=2,
        cache_resolution=16,
        clip_stem="source",
    )

    assert extract_calls == ["source_clip_0.mp4", "source_clip_1.mp4", "source_clip_2.mp4"]
    assert [clip["cached_video"] for clip in second] == [
        clip["cached_video"] for clip in first
    ]


def test_materialize_dataset_cached_videos_rewrites_virtual_entries(tmp_path, monkeypatch):
    pytest.importorskip("decord")
    from trace_tad import data_prep

    video = tmp_path / "source.mp4"
    _write_tiny_video(video, frame_count=4, fps=2.0)
    data_prep._load_or_build_pts(os.fspath(video))
    annotation = tmp_path / "dataset.json"
    annotation.write_text(json.dumps({
        "database": {
            "source_clip_0": {
                "subset": "validation",
                "frame": 2,
                "duration": 1.0,
                "annotations": [],
                "source_video": os.fspath(video),
                "source_frame_offset": 0,
                "source_pts_table": os.fspath(video) + ".pts.npy",
            }
        }
    }), encoding="utf-8")

    def fake_extract(_video_path, clip_path, *_args, **_kwargs):
        extract_calls.append(Path(clip_path).name)
        Path(clip_path).parent.mkdir(parents=True, exist_ok=True)
        Path(clip_path).write_bytes(b"clip")
        return True, None

    extract_calls = []
    monkeypatch.setattr(data_prep, "_extract_cached_video_clip", fake_extract)

    cached_json = data_prep.materialize_dataset_cached_videos(
        os.fspath(annotation),
        os.fspath(tmp_path / "eval"),
        clip_frames=2,
        cache_resolution=16,
    )

    payload = json.loads(Path(cached_json).read_text(encoding="utf-8"))
    entry = payload["database"]["source_clip_0"]
    assert Path(entry["cached_video"]).is_file()
    cache_dir = tmp_path / "source.mp4.trace-clips" / "f2_r16_crf23"
    assert Path(entry["cached_video"]).parent == cache_dir
    assert (cache_dir / "manifest.json").is_file()
    assert extract_calls == ["source_clip_0.mp4"]

    reused_json = data_prep.materialize_dataset_cached_videos(
        cached_json,
        os.fspath(tmp_path / "eval"),
        clip_frames=2,
        cache_resolution=16,
    )
    assert reused_json == cached_json
    assert extract_calls == ["source_clip_0.mp4"]

    stat = video.stat()
    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    data_prep.materialize_dataset_cached_videos(
        cached_json,
        os.fspath(tmp_path / "eval"),
        clip_frames=2,
        cache_resolution=16,
    )
    assert extract_calls == ["source_clip_0.mp4", "source_clip_0.mp4"]


def test_find_video_csv_pairs_filters_by_included_stems(tmp_path):
    from trace_tad.data_prep import _find_video_csv_pairs

    for stem in ("alpha", "beta", "gamma"):
        _touch_pair(tmp_path, stem)
    # Bonus: a remux copy should be reachable via the source stem
    (tmp_path / "alpha.remux.mp4").write_bytes(b"")

    all_pairs = _find_video_csv_pairs(str(tmp_path))
    assert {Path(v).stem for v, _ in all_pairs} >= {"alpha", "beta", "gamma"}

    filtered = _find_video_csv_pairs(str(tmp_path), included_stems={"alpha", "gamma"})
    stems = {Path(v).stem.replace(".remux", "") for v, _ in filtered}
    assert stems == {"alpha", "gamma"}, stems

    # Empty allowlist is treated like None (no filter) — frontend validation
    # blocks empty submissions before they reach the CLI, so this avoids
    # surprising "matched no pairs" failures from a stale config.
    assert _find_video_csv_pairs(str(tmp_path), included_stems=set()) == all_pairs


def test_pair_stem_matching_uses_frontend_grouping_rule(tmp_path):
    from trace_tad.data_prep import _find_video_csv_pairs

    # Frontend PathPicker groups `trial.mkv`, `trial.mkv.remux.mp4`, and
    # `trial.csv` under the single key `trial`.
    (tmp_path / "trial.mkv").write_bytes(b"")
    (tmp_path / "trial.mkv.remux.mp4").write_bytes(b"")
    (tmp_path / "trial.csv").write_text("labelId\n", encoding="utf-8")

    pairs = _find_video_csv_pairs(str(tmp_path), included_stems={"trial"})

    assert pairs == [(str(tmp_path / "trial.mkv"), str(tmp_path / "trial.csv"))]


def test_explicit_pair_spec_selects_exact_video_and_csv(tmp_path):
    from trace_tad.data_prep import _find_video_csv_pairs

    (tmp_path / "trial.mp4").write_bytes(b"")
    (tmp_path / "other.mp4").write_bytes(b"")
    (tmp_path / "trial.csv").write_text("labelId\ncanonical\n", encoding="utf-8")
    (tmp_path / "trial_manual.csv").write_text("labelId\nmanual\n", encoding="utf-8")

    pairs = _find_video_csv_pairs(
        str(tmp_path),
        explicit_pairs=["trial.mp4=trial_manual.csv"],
    )

    assert pairs == [(str(tmp_path / "trial.mp4"), str(tmp_path / "trial_manual.csv"))]


def test_extract_classes_skips_trace_meta_comment(tmp_path):
    from trace_tad.data_prep import _extract_classes_from_csvs

    csv_path = tmp_path / "trial.csv"
    csv_path.write_text(
        "# trace-meta: behaviors=shepherd:s\n"
        "labelId,timestamp,endTimestamp\n"
        "shepherd,0.0,1.0\n",
        encoding="utf-8",
    )

    assert _extract_classes_from_csvs([str(csv_path)]) == ["shepherd"]


def test_explicit_pair_spec_errors_for_missing_csv(tmp_path):
    from trace_tad.data_prep import _find_video_csv_pairs

    (tmp_path / "trial.mp4").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="Pair CSV not found"):
        _find_video_csv_pairs(str(tmp_path), explicit_pairs=["trial.mp4=missing.csv"])


def test_prepare_dataset_with_selected_pairs_uses_selection_specific_cache(monkeypatch, tmp_path):
    from trace_tad import data_prep

    for stem in ("alpha", "beta"):
        (tmp_path / f"{stem}.mp4").write_bytes(b"")
        (tmp_path / f"{stem}.csv").write_text(
            "labelId,timestamp,endTimestamp\n"
            f"{stem}_label,0.0,1.0\n",
            encoding="utf-8",
        )

    def fake_process_video(video_path, csv_path, output_dir, clip_frames=768, virtual_clips=True, **kwargs):
        stem = Path(video_path).stem
        return {
            0: {
                "duration": 1.0,
                "frame": 30,
                "annotations": [{
                    "frame_segment": [0, 29],
                    "segment": [0.0, 1.0],
                    "timestamp_sec": [0.0, 1.0],
                    "label": f"{stem}_label",
                }],
                "source_video": os.path.abspath(video_path),
                "source_frame_offset": 0,
            }
        }

    monkeypatch.setattr(data_prep, "_process_video", fake_process_video)

    model_dir, json_path, classmap_path = data_prep.prepare_dataset(
        str(tmp_path),
        included_stems=["beta"],
    )

    assert Path(model_dir).name.startswith("model_")
    assert Path(json_path).parent == Path(model_dir)
    assert Path(classmap_path).parent == Path(model_dir)

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert list(payload["database"]) == ["beta_clip_0"]
    assert Path(classmap_path).read_text(encoding="utf-8").splitlines() == ["beta_label"]

    default_model_dir, _, _ = data_prep.prepare_dataset(str(tmp_path))
    assert Path(default_model_dir).name.startswith("model_")


def test_prepare_dataset_with_explicit_pair_csv_uses_selected_annotation(monkeypatch, tmp_path):
    from trace_tad import data_prep

    (tmp_path / "beta.mp4").write_bytes(b"")
    (tmp_path / "beta.csv").write_text(
        "labelId,timestamp,endTimestamp\ncanonical,0.0,1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "beta_manual.csv").write_text(
        "labelId,timestamp,endTimestamp\nmanual,0.0,1.0\n",
        encoding="utf-8",
    )

    def fake_process_video(video_path, csv_path, output_dir, clip_frames=768, virtual_clips=True, **kwargs):
        label = Path(csv_path).stem
        return {
            0: {
                "duration": 1.0,
                "frame": 30,
                "annotations": [{
                    "frame_segment": [0, 29],
                    "segment": [0.0, 1.0],
                    "timestamp_sec": [0.0, 1.0],
                    "label": label,
                }],
                "source_video": os.path.abspath(video_path),
                "source_frame_offset": 0,
            }
        }

    monkeypatch.setattr(data_prep, "_process_video", fake_process_video)

    _, json_path, classmap_path = data_prep.prepare_dataset(
        str(tmp_path),
        explicit_pairs=["beta.mp4=beta_manual.csv"],
    )

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    entry = payload["database"]["beta_clip_0"]
    assert entry["annotations"][0]["label"] == "beta_manual"
    assert Path(classmap_path).read_text(encoding="utf-8").splitlines() == ["manual"]


def test_prep_dataset_script_writes_result_for_model_dir(tmp_path):
    dataset = tmp_path / "raw"
    dataset.mkdir()
    _write_tiny_video(dataset / "trial.mp4")
    (dataset / "trial.csv").write_text(
        "labelId,timestamp,endTimestamp\n"
        "walk,0.0,1.0\n",
        encoding="utf-8",
    )
    output = tmp_path / "prep_result.json"
    model_dir = tmp_path / "model_20260507_120000"

    result = subprocess.run(
        [
            sys.executable,
            "tools/prep_dataset.py",
            str(dataset),
            "--output-dir",
            str(model_dir),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Prep result saved to" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert Path(payload["model_dir"]).resolve() == model_dir.resolve()
    assert Path(payload["dataset_json"]).resolve() == (model_dir / "dataset.json").resolve()
    assert Path(payload["classmap_path"]).resolve() == (model_dir / "classmap.txt").resolve()


def test_prepare_dataset_cached_video_writes_cache_and_preserves_source_metadata(tmp_path):
    from trace_tad import data_prep

    _write_tiny_video(tmp_path / "trial.mp4", frame_count=12, fps=3.0)
    (tmp_path / "trial.csv").write_text(
        "labelId,timestamp,endTimestamp\n"
        "walk,0.0,1.5\n",
        encoding="utf-8",
    )
    model_dir = tmp_path / "model_20260508_010000"

    _, json_path, _ = data_prep.prepare_dataset(
        str(tmp_path),
        clip_frames=6,
        output_dir=str(model_dir),
        cache_mode="cached_video",
        cache_resolution=16,
    )

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    entry = payload["database"]["trial_clip_0"]
    cache_dir = tmp_path / "trial.mp4.trace-clips" / "f6_r16_crf23"
    assert Path(entry["cached_video"]).is_file()
    assert Path(entry["cached_video"]).parent == cache_dir
    assert (cache_dir / "manifest.json").is_file()
    assert entry["source_video"] == str((tmp_path / "trial.mp4").resolve())
    assert entry["source_frame_offset"] == 0


def test_prepare_video_info_prefers_cached_video_but_virtual_keeps_source_offset():
    from trace_tad.datasets.transforms.end_to_end import PrepareVideoInfo
    from trace_tad.datasets.transforms.video_transforms import VideoDecode
    np = pytest.importorskip("numpy")

    cached = PrepareVideoInfo()({
        "video_name": "clip",
        "data_path": "/data",
        "cached_video": "/cache/clip.mp4",
        "source_video": "/raw/source.mkv",
        "source_frame_offset": 768,
    })
    assert cached["filename"] == "/cache/clip.mp4"
    assert cached["decode_frame_offset"] == 0
    assert cached["source_frame_offset"] == 768

    virtual = PrepareVideoInfo()({
        "video_name": "clip",
        "data_path": "/data",
        "source_video": "/raw/source.mkv",
        "source_frame_offset": 768,
    })
    assert virtual["filename"] == "/raw/source.mkv"
    assert "decode_frame_offset" not in virtual

    class Batch:
        def __init__(self, count):
            self.count = count

        def asnumpy(self):
            return np.zeros((self.count, 1, 1, 3), dtype=np.uint8)

    class Reader:
        def __init__(self):
            self.indices = None

        def __len__(self):
            return 1000

        def get_batch(self, indices):
            self.indices = indices
            return Batch(len(indices))

    reader = Reader()
    VideoDecode()({
        "frame_inds": np.array([0, 1]),
        "video_reader": reader,
        "source_frame_offset": 10,
        "decode_frame_offset": 0,
    })
    assert reader.indices == [0, 1]

    reader = Reader()
    VideoDecode()({
        "frame_inds": np.array([0, 1]),
        "video_reader": reader,
        "source_frame_offset": 10,
    })
    assert reader.indices == [10, 11]
