import shutil
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import app


client = TestClient(app)


ROOT = Path(__file__).resolve().parents[1]
DEV_CFR_VIDEO = ROOT / "data" / "dev_test" / "clips" / "2025-06-27_14_51_47_clip_158.mp4"


def test_list_dirs_returns_children_for_existing_directory_without_trailing_slash(tmp_path):
    parent = tmp_path / "datasets"
    alpha = parent / "alpha"
    beta = parent / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir()

    response = client.get("/api/dirs", params={"prefix": str(parent)})

    assert response.status_code == 200
    returned = response.json()["dirs"]
    assert str(alpha) in returned
    assert str(beta) in returned


def test_list_paths_returns_matching_directories_and_filtered_files(tmp_path):
    root = tmp_path / "videos"
    root.mkdir()
    matching_dir = root / "video_folder"
    matching_dir.mkdir()
    mp4 = root / "video_clip.mp4"
    avi = root / "video_alt.avi"
    txt = root / "video_notes.txt"
    mp4.write_bytes(b"mp4")
    avi.write_bytes(b"avi")
    txt.write_text("ignore me")

    response = client.get(
        "/api/paths",
        params={
            "prefix": str(root / "video"),
            "extensions": ".mp4,.avi",
        },
    )

    assert response.status_code == 200
    returned = response.json()["paths"]
    assert {"path": str(matching_dir), "type": "dir"} in returned
    assert {"path": str(mp4), "type": "file"} in returned
    assert {"path": str(avi), "type": "file"} in returned
    assert {"path": str(txt), "type": "file"} not in returned


def test_upload_videos_saves_csv_companions(tmp_path):
    response = client.post(
        "/api/upload-videos",
        data={"destination": str(tmp_path)},
        files=[
            ("files", ("movie.mp4", BytesIO(b"video bytes"), "video/mp4")),
            ("files", ("movie.csv", BytesIO(b"labelId,timestamp,endTimestamp\n"), "text/csv")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["files"] == ["movie.mp4"]
    assert {"originalName": "movie.mp4", "savedName": "movie.mp4", "type": "video"} in payload["uploaded"]
    assert {"originalName": "movie.csv", "savedName": "movie.csv", "type": "csv"} in payload["uploaded"]
    assert (tmp_path / "movie.mp4").exists()
    assert (tmp_path / "movie.csv").exists()


# ─────────────────────────────────────────────────────────────────────
# Phase 4 of docs/pts-based-frame-mapping.md — frame-rate metadata is
# threaded from ffprobe through `_get_video_info`, into the
# `/api/files` and `/api/files/{filename}/probe` JSON responses, so the
# annotator frontend can drive its frame-step UI off the actual avg fps
# instead of the legacy hard-coded 30, and show a VFR badge in the
# file browser.
# ─────────────────────────────────────────────────────────────────────


def test_parse_rational_handles_edge_cases():
    from trace_tad.server.app import _parse_rational

    assert _parse_rational("30/1") == 30.0
    # Real NTSC fraction.
    assert _parse_rational("30000/1001") == pytest.approx(29.97, abs=1e-3)
    # Single-number form (some containers report this).
    assert _parse_rational("29.97") == 29.97
    # Pathological inputs must return None, not raise / inf / NaN.
    assert _parse_rational("0/0") is None
    assert _parse_rational("") is None
    assert _parse_rational("garbage") is None
    assert _parse_rational(None) is None  # type: ignore[arg-type]


@pytest.mark.skipif(
    not DEV_CFR_VIDEO.is_file(),
    reason=f"requires CFR fixture {DEV_CFR_VIDEO} (dev_test dataset)",
)
def test_get_video_info_returns_frame_rate_metadata_for_cfr_clip():
    from trace_tad.server.app import _get_video_info

    info = _get_video_info(str(DEV_CFR_VIDEO))
    assert info is not None
    assert info["codec"] == "h264"
    assert info["container"] == "mp4"
    # Frame-rate fields exist and are sensible. The dev_test clip is a
    # clean 30-fps CFR file, so r_frame_rate ≈ avg_frame_rate ≈ 30.
    assert info["rFrameRate"] == pytest.approx(30.0, abs=0.1)
    assert info["avgFrameRate"] == pytest.approx(30.0, abs=0.1)
    # CFR ⇒ isVfr is False (not None and not True).
    assert info["isVfr"] is False


@pytest.mark.skipif(
    not DEV_CFR_VIDEO.is_file(),
    reason=f"requires CFR fixture {DEV_CFR_VIDEO} (dev_test dataset)",
)
def test_list_files_exposes_frame_rate_fields(tmp_path):
    """`/api/files` must propagate the new frame-rate metadata into each
    `filesInfo` entry so the FileBrowser can drive its codec / VFR
    badges and the Editor can use the real fps."""
    shutil.copy(DEV_CFR_VIDEO, tmp_path / "clip.mp4")

    response = client.get("/api/files", params={"dir": str(tmp_path)})
    assert response.status_code == 200
    payload = response.json()

    assert payload["files"] == ["clip.mp4"]
    assert len(payload["filesInfo"]) == 1
    fi = payload["filesInfo"][0]

    # Existing fields still present (regression guard).
    assert fi["name"] == "clip.mp4"
    assert fi["codec"] == "h264"
    assert fi["isH264"] is True

    # New Phase 4 fields are present and correct.
    assert "rFrameRate" in fi
    assert "avgFrameRate" in fi
    assert "isVfr" in fi
    assert fi["rFrameRate"] == pytest.approx(30.0, abs=0.1)
    assert fi["avgFrameRate"] == pytest.approx(30.0, abs=0.1)
    assert fi["isVfr"] is False


@pytest.mark.skipif(
    not DEV_CFR_VIDEO.is_file(),
    reason=f"requires CFR fixture {DEV_CFR_VIDEO} (dev_test dataset)",
)
def test_probe_endpoint_exposes_frame_rate_fields(tmp_path):
    """`/api/files/{filename}/probe` carries the same frame-rate
    metadata so single-file consumers don't need to list a directory."""
    shutil.copy(DEV_CFR_VIDEO, tmp_path / "clip.mp4")

    response = client.get(
        "/api/files/clip.mp4/probe", params={"dir": str(tmp_path)}
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["codec"] == "h264"
    assert payload["needsTranscode"] is False
    assert payload["rFrameRate"] == pytest.approx(30.0, abs=0.1)
    assert payload["avgFrameRate"] == pytest.approx(30.0, abs=0.1)
    assert payload["isVfr"] is False


@pytest.mark.skipif(
    not DEV_CFR_VIDEO.is_file(),
    reason=f"requires CFR fixture {DEV_CFR_VIDEO} (dev_test dataset)",
)
def test_pts_endpoint_returns_float32_array_and_caches_to_disk(tmp_path):
    """`/api/files/{filename}/pts` builds the per-frame PTS index on first
    hit and returns it as a binary Float32 array. The annotator uses this
    to snap behavior-interval boundaries to real frames on VFR clips
    (where ``time = frame / fps`` lies). Subsequent calls must hit the
    on-disk ``.pts.npy`` cache instead of re-decoding."""
    import numpy as np

    shutil.copy(DEV_CFR_VIDEO, tmp_path / "clip.mp4")

    response = client.get(
        "/api/files/clip.mp4/pts", params={"dir": str(tmp_path)}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"

    frame_count = int(response.headers["x-frame-count"])
    assert frame_count > 0
    # Float32 → 4 bytes per frame; size must match the header.
    assert len(response.content) == frame_count * 4

    pts = np.frombuffer(response.content, dtype=np.float32)
    # Strictly monotonically increasing — that's the whole point of PTS.
    assert (np.diff(pts) > 0).all()
    # First frame at t=0 (or very close); CFR fixture ⇒ ~30 fps spacing.
    assert pts[0] == pytest.approx(0.0, abs=1e-3)
    assert (pts[1] - pts[0]) == pytest.approx(1 / 30, abs=1e-3)

    # The cache file must now sit next to the source for future readers
    # (training pipeline, second annotator session, etc.).
    cache = tmp_path / "clip.mp4.pts.npy"
    assert cache.is_file()

    # Second request must succeed (and reuse the cache) without rebuilding.
    response2 = client.get(
        "/api/files/clip.mp4/pts", params={"dir": str(tmp_path)}
    )
    assert response2.status_code == 200
    assert response2.content == response.content


def test_pts_endpoint_returns_404_for_missing_file(tmp_path):
    response = client.get(
        "/api/files/does_not_exist.mp4/pts", params={"dir": str(tmp_path)}
    )
    assert response.status_code == 404


def test_list_files_returns_null_frame_rate_when_ffprobe_fails(tmp_path, monkeypatch):
    """Files that can't be probed (corrupt / non-video bytes) must
    still appear in the response, with the new fields set to ``null``
    rather than crashing or producing partial entries."""
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a real video file")

    response = client.get("/api/files", params={"dir": str(tmp_path)})
    assert response.status_code == 200
    payload = response.json()

    assert payload["files"] == ["broken.mp4"]
    fi = payload["filesInfo"][0]
    # ffprobe failed → codec is None and the new fields are null too.
    assert fi["codec"] is None
    assert fi["rFrameRate"] is None
    assert fi["avgFrameRate"] is None
    assert fi["isVfr"] is None


# ── Multi-CSV + shortcut metadata round-trip ───────────────────────────

def test_save_labels_writes_trace_meta_line_when_shortcuts_provided(tmp_path):
    """Behaviors with shortcuts should round-trip through the CSV via the
    leading `# trace-meta:` line, so re-opening the file restores the
    keyboard bindings instead of forcing the user to redo them."""
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"not a real video")  # ffprobe failure is fine here.

    response = client.post(
        "/api/files/movie.mp4/labels",
        params={"dir": str(tmp_path)},
        json={
            "videoPath": str(tmp_path),
            "labelRects": [
                {"behavior": "sniffing", "timestamp": 0.5, "endTimestamp": 1.5},
                {"behavior": "nursing",  "timestamp": 2.0, "endTimestamp": 3.0},
            ],
            "behaviors": [
                {"name": "sniffing", "shortcut": "s"},
                {"name": "nursing",  "shortcut": "n"},
            ],
        },
    )
    assert response.status_code == 200

    csv_text = (tmp_path / "movie.csv").read_text()
    first_line = csv_text.splitlines()[0]
    assert first_line.startswith("# trace-meta: behaviors=")
    # Order follows the payload; URL-encoded for safety.
    assert "sniffing:s" in first_line
    assert "nursing:n" in first_line


def test_save_labels_omits_meta_line_when_no_shortcuts(tmp_path):
    """A payload without `behaviors` (or with empty shortcuts) must not
    write a metadata line — older readers expecting a plain header in
    line 1 stay happy."""
    response = client.post(
        "/api/files/movie.mp4/labels",
        params={"dir": str(tmp_path)},
        json={
            "videoPath": str(tmp_path),
            "labelRects": [
                {"behavior": "walking", "timestamp": 0.1, "endTimestamp": 0.9},
            ],
        },
    )
    assert response.status_code == 200

    csv_text = (tmp_path / "movie.csv").read_text()
    assert not csv_text.startswith("#")
    assert csv_text.splitlines()[0] == "labelId,timestamp,endTimestamp"


def test_save_labels_writes_to_specific_csv_name(tmp_path):
    """`csvName` query param routes the write to a specific file so the
    multi-CSV picker in the sidebar (rater A vs rater B) works."""
    response = client.post(
        "/api/files/movie.mp4/labels",
        params={"dir": str(tmp_path), "csvName": "movie_v2.csv"},
        json={
            "videoPath": str(tmp_path),
            "labelRects": [
                {"behavior": "walking", "timestamp": 0.0, "endTimestamp": 1.0},
            ],
        },
    )
    assert response.status_code == 200
    assert (tmp_path / "movie_v2.csv").exists()
    assert not (tmp_path / "movie.csv").exists()


def test_save_labels_rejects_unsafe_csv_name(tmp_path):
    """csvName must belong to the video and never escape the directory."""
    bad = client.post(
        "/api/files/movie.mp4/labels",
        params={"dir": str(tmp_path), "csvName": "../escape.csv"},
        json={"videoPath": str(tmp_path), "labelRects": []},
    )
    assert bad.status_code == 400


def test_delete_csv_removes_specific_csv_name(tmp_path):
    """DELETE /csv removes only the requested annotation variant."""
    (tmp_path / "movie.csv").write_text("labelId,timestamp,endTimestamp\n")
    (tmp_path / "movie_v2.csv").write_text("labelId,timestamp,endTimestamp\n")

    response = client.delete(
        "/api/files/movie.mp4/csv",
        params={"dir": str(tmp_path), "csvName": "movie_v2.csv"},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "movie_v2.csv"
    assert (tmp_path / "movie.csv").exists()
    assert not (tmp_path / "movie_v2.csv").exists()


def test_delete_csv_rejects_unsafe_csv_name(tmp_path):
    bad = client.delete(
        "/api/files/movie.mp4/csv",
        params={"dir": str(tmp_path), "csvName": "../escape.csv"},
    )

    assert bad.status_code == 400


def test_list_files_exposes_csv_files_per_video(tmp_path):
    """`/api/files` returns the discovered CSVs for each video so the
    sidebar can render a row per labeling."""
    (tmp_path / "movie.mp4").write_bytes(b"x")
    # Canonical CSV plus two raters' variants.
    (tmp_path / "movie.csv").write_text("labelId,timestamp,endTimestamp\n")
    (tmp_path / "movie_raterA.csv").write_text("labelId,timestamp,endTimestamp\n")
    (tmp_path / "movie_raterB.csv").write_text("labelId,timestamp,endTimestamp\n")
    # An unrelated CSV that must NOT bleed into this video's bucket
    # (different base name — `movie10` is a separate video conceptually).
    (tmp_path / "movie10.csv").write_text("labelId,timestamp,endTimestamp\n")

    response = client.get("/api/files", params={"dir": str(tmp_path)})
    assert response.status_code == 200
    fi = response.json()["filesInfo"][0]

    assert fi["hasCsv"] is True
    assert fi["csvFiles"][0] == "movie.csv"  # canonical first
    assert set(fi["csvFiles"]) == {"movie.csv", "movie_raterA.csv", "movie_raterB.csv"}
    assert "movie10.csv" not in fi["csvFiles"]
