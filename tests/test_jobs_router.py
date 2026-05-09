import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import trace_tad.server.jobs_router as jobs_router
from trace_tad.jobs.models import JobInfo, JobStatus, JobType


def _build_client():
    app = FastAPI()
    app.include_router(jobs_router.router)
    return TestClient(app)


def test_list_models_reads_config_and_class_metadata(tmp_path, monkeypatch):
    root = tmp_path
    (root / "tools").mkdir()
    (root / "tools" / "train.py").write_text("# marker\n", encoding="utf-8")

    model_dir = root / "data" / "dev_suite" / "model_20260507_120000"
    model_dir.mkdir(parents=True)
    (model_dir / "best.pth").write_bytes(b"x" * 4096)
    (model_dir / "classmap.txt").write_text("drink\neat\nrear\n", encoding="utf-8")
    (model_dir / "config.txt").write_text("configs/large.py\n", encoding="utf-8")

    ignored_dir = root / "data" / "broken" / "model_20260507_120000"
    ignored_dir.mkdir(parents=True)
    (ignored_dir / "classmap.txt").write_text("missing checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(jobs_router, "_find_project_root", lambda: root)
    client = _build_client()

    response = client.get("/api/models")

    assert response.status_code == 200
    models = response.json()
    assert len(models) == 1
    assert models[0]["path"] == str(model_dir)
    assert models[0]["label"] == "data/dev_suite/model_20260507_120000"
    assert models[0]["config_path"] == "configs/large.py"
    assert models[0]["classes"] == ["drink", "eat", "rear"]
    assert models[0]["num_classes"] == 3
    assert models[0]["size_mb"] >= 0


def test_job_artifact_endpoint_returns_newest_match_and_blocks_path_traversal(tmp_path, monkeypatch):
    older = tmp_path / "run_1" / "metrics.json"
    newer = tmp_path / "run_2" / "metrics.json"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text('{"source":"older"}', encoding="utf-8")
    newer.write_text('{"source":"newer"}', encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    job = JobInfo(
        job_id="job-123",
        job_type=JobType.TEST,
        status=JobStatus.COMPLETED,
        config_path="configs/small.py",
        created_at="2026-04-01T00:00:00Z",
        log_file=str(tmp_path / "job.log"),
        work_dir=str(tmp_path),
    )

    class FakeManager:
        def get_job(self, job_id):
            assert job_id == "job-123"
            return job

    monkeypatch.setattr(jobs_router, "manager", FakeManager())
    client = _build_client()

    response = client.get("/api/jobs/job-123/artifacts/metrics.json")

    assert response.status_code == 200
    assert response.json() == {"source": "newer"}

    invalid = client.get("/api/jobs/job-123/artifacts/..%5Csecret.txt")
    assert invalid.status_code == 400


def test_submit_train_tune_job_uses_manager(tmp_path, monkeypatch):
    job = JobInfo(
        job_id="tune-123",
        job_type=JobType.TRAIN_TUNE,
        status=JobStatus.QUEUED,
        config_path="configs/small.py",
        created_at="2026-04-01T00:00:00Z",
        log_file=str(tmp_path / "train_tune.log"),
        work_dir=str(tmp_path),
    )

    class FakeManager:
        def start_train_tune_job(self, request):
            assert request.model_dir == str(tmp_path)
            assert request.annotation_path.endswith("dataset.json")
            return job

    monkeypatch.setattr(jobs_router, "manager", FakeManager())
    client = _build_client()

    response = client.post("/api/jobs/train-tune", json={
        "config_path": "configs/small.py",
        "model_dir": str(tmp_path),
        "annotation_path": str(tmp_path / "dataset.json"),
        "class_map": str(tmp_path / "classmap.txt"),
    })

    assert response.status_code == 202
    assert response.json()["job_id"] == "tune-123"


def test_pipeline_command_endpoint_returns_command():
    client = _build_client()

    response = client.post("/api/pipeline/command", json={
        "steps": {"train": True, "extra_test": False, "infer": True},
        "train_selection": {
            "folder": "/datasets/train set",
            "pairs": ["/datasets/train set/a.mp4=/datasets/train set/a.csv"],
        },
        "input_selection": {
            "folder": "/videos",
            "stems": ["clip01"],
        },
        "model_size": "large",
        "cache_mode": "cached_video",
        "cache_resolution": 192,
        "resource_profile": "auto",
        "epochs": 100,
        "val_start_epoch": 50,
        "val_interval": 10,
        "train_ratio": 0.8,
    })

    assert response.status_code == 200
    body = response.json()
    assert body["argv"][:5] == ["trace", "pipeline", "--model", "large", "--train"]
    assert "--infer" in body["argv"]
    assert body["argv"][body["argv"].index("--cache-resolution") + 1] == "192"
    assert "'/datasets/train set'" in body["command"]
    assert body["warnings"] == []


def test_pipeline_command_endpoint_returns_validation_error():
    client = _build_client()

    response = client.post("/api/pipeline/command", json={
        "steps": {"train": True, "extra_test": False, "infer": False},
    })

    assert response.status_code == 400
    assert response.json()["detail"] == "Pick at least one training pair."
