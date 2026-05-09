from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app


def _write_pair(tmp_path: Path):
    video = tmp_path / "session01.mp4"
    csv = tmp_path / "session01.csv"
    video.write_bytes(b"video")
    csv.write_text(
        "# trace-meta: behaviors=groom:g,rear:r\n"
        "labelId,timestamp,endTimestamp\n"
        "groom,0.0,2.0\n"
        "rear,40.0,42.0\n"
        "groom,70.0,72.0\n",
        encoding="utf-8",
    )
    return video, csv


def test_training_resource_estimate_counts_behaviors_and_cache_recommendation(monkeypatch, tmp_path):
    from trace_tad import training_resources

    _write_pair(tmp_path)

    def fake_probe(video_path):
        return {
            "path": video_path,
            "size_bytes": 5 * 1024 * 1024 * 1024,
            "duration_sec": 7200.0,
            "fps": 30.0,
            "frame_count": 216000,
            "width": 1920,
            "height": 1080,
        }

    monkeypatch.setattr(training_resources, "_probe_video", fake_probe)

    estimate = training_resources.estimate_training_resources(
        work_dir=str(tmp_path),
        explicit_pairs=["session01.mp4=session01.csv"],
        clip_frames=768,
        cache_resolution=144,
    )

    assert estimate["summary"]["pair_count"] == 1
    assert estimate["summary"]["annotation_count"] == 3
    assert estimate["behaviors"] == [
        {"name": "groom", "count": 2},
        {"name": "rear", "count": 1},
    ]
    assert estimate["recommendations"]["cache_mode"] == "cached_video"
    assert any(option["recommended"] for option in estimate["cache_options"])
    assert estimate["model_options"][0]["vram_mb"] > 0
    assert estimate["resolution_options"][1]["id"] == 144
    assert estimate["resolution_options"][1]["small_vram_mb"] > estimate["model_options"][0]["vram_mb"]
    assert estimate["resolution_options"][3]["small_vram_mb"] > estimate["resolution_options"][0]["small_vram_mb"]
    assert estimate["resource_profiles"][0]["ram_mb"] > 0


def test_training_estimate_endpoint_returns_400_for_missing_pair(tmp_path):
    client = TestClient(app)

    response = client.post(
        "/api/training/estimate",
        json={
            "work_dir": str(tmp_path),
            "explicit_pairs": ["missing.mp4=missing.csv"],
        },
    )

    assert response.status_code == 400
    assert "Pair video not found" in response.json()["detail"]
