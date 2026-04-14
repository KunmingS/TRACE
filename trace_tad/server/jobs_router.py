"""FastAPI router for TRACE training, testing, and inference jobs."""
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from trace_tad.jobs import (
    JobManager,
    JobInfo,
    JobStatus,
    JobType,
    TrainRequest,
    TestRequest,
    InferRequest,
    PrepRequest,
)

router = APIRouter()

# Module-level singleton job manager
manager = JobManager(max_concurrency=1)


def _find_project_root():
    """Locate the TRACE project root."""
    # Check relative to this file (trace_tad/server/jobs_router.py -> trace_tad -> TRACE/)
    pkg_dir = Path(__file__).resolve().parent.parent.parent
    if (pkg_dir / "tools" / "train.py").is_file():
        return pkg_dir
    cwd = Path.cwd()
    if (cwd / "tools" / "train.py").is_file():
        return cwd
    return None


# ── Job Endpoints ───────────────────────────────────────────────────


@router.post("/api/jobs/train", status_code=202, response_model=JobInfo)
async def submit_train_job(request: TrainRequest):
    """Submit a training job to the queue."""
    job = manager.start_train_job(request)
    return job


@router.post("/api/jobs/test", status_code=202, response_model=JobInfo)
async def submit_test_job(request: TestRequest):
    """Submit a test/evaluation job to the queue."""
    job = manager.start_test_job(request)
    return job


@router.post("/api/jobs/infer", status_code=202, response_model=JobInfo)
async def submit_infer_job(request: InferRequest):
    """Submit an inference job to the queue (no annotations needed)."""
    job = manager.start_infer_job(request)
    return job


@router.post("/api/jobs/prep", status_code=202, response_model=JobInfo)
async def submit_prep_job(request: PrepRequest):
    """Submit a dataset preparation job (clip videos + generate annotations)."""
    job = manager.start_prep_job(request)
    return job


@router.get("/api/jobs", response_model=list[JobInfo])
async def list_jobs(
    type: Optional[str] = Query(None, description="Filter by job type (train/test/infer/prep)"),
    status: Optional[str] = Query(None, description="Filter by status"),
):
    """List all jobs, optionally filtered by type and/or status."""
    type_filter = JobType(type) if type else None
    status_filter = JobStatus(status) if status else None
    return manager.list_jobs(type_filter=type_filter, status_filter=status_filter)


@router.get("/api/jobs/{job_id}", response_model=JobInfo)
async def get_job(job_id: str):
    """Get details of a single job."""
    try:
        return manager.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@router.get("/api/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    n: int = Query(100, description="Number of lines to return"),
):
    """Get the last N lines of a job's log."""
    try:
        lines = manager.get_log_tail(job_id, n_lines=n)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"lines": lines}


@router.get("/api/jobs/{job_id}/logs/stream")
async def stream_job_logs(job_id: str):
    """Stream job logs via Server-Sent Events."""
    try:
        manager.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    def event_generator():
        for line in manager.stream_log(job_id):
            yield f"data: {line}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/jobs/{job_id}/cancel", response_model=JobInfo)
async def cancel_job(job_id: str):
    """Cancel a queued or running job."""
    try:
        return manager.cancel_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/jobs/{job_id}/artifacts/{filename}")
async def get_job_artifact(job_id: str, filename: str):
    """Serve a file from a job's work directory (e.g., metrics.json, predictions.json)."""
    try:
        job = manager.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if not job.work_dir:
        raise HTTPException(status_code=404, detail="Job has no work directory")

    # Security: only allow simple filenames (no path traversal)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Search for the artifact in the work_dir tree
    # Artifacts may be in work_dir directly or in experiment subdirectories
    root = Path(job.work_dir)
    candidates = list(root.rglob(filename))
    if not candidates:
        raise HTTPException(status_code=404, detail=f"Artifact '{filename}' not found")

    # Return the most recently modified match
    artifact = max(candidates, key=lambda p: p.stat().st_mtime)
    return FileResponse(str(artifact), filename=filename)


# ── Config and Checkpoint Discovery ─────────────────────────────────


@router.get("/api/configs")
async def list_configs():
    """List available config files."""
    root = _find_project_root()
    if root is None:
        raise HTTPException(status_code=500, detail="Cannot find project root")

    configs_dir = root / "configs"
    if not configs_dir.is_dir():
        return []

    configs = []
    for py_file in sorted(configs_dir.rglob("*.py")):
        rel = py_file.relative_to(root)
        name = rel.name
        # Skip base/internal configs
        if name.startswith("_"):
            continue
        configs.append({
            "path": str(rel).replace("\\", "/"),
            "name": name.removesuffix(".py"),
        })
    return configs


@router.get("/api/checkpoints")
async def list_checkpoints():
    """List available checkpoints in experiment directories."""
    root = _find_project_root()
    if root is None:
        raise HTTPException(status_code=500, detail="Cannot find project root")

    exps_dir = root / "exps"
    if not exps_dir.is_dir():
        return []

    checkpoints = []
    for pth_file in sorted(exps_dir.rglob("*.pth")):
        rel = pth_file.relative_to(root)
        size_mb = pth_file.stat().st_size / (1024 * 1024)
        checkpoints.append({
            "path": str(rel).replace("\\", "/"),
            "name": pth_file.name,
            "experiment": pth_file.parent.parent.name if pth_file.parent.name == "checkpoint" else pth_file.parent.name,
            "size_mb": round(size_mb, 1),
        })
    return checkpoints


@router.get("/api/models")
async def list_models():
    """List available model folders (containing best.pth + classmap.txt).

    Scans ./model/ and exps/*/model/ for valid model directories.
    """
    root = _find_project_root()
    if root is None:
        raise HTTPException(status_code=500, detail="Cannot find project root")

    models = []

    def _scan_model_dir(model_dir: Path, label: str):
        best_pth = model_dir / "best.pth"
        classmap = model_dir / "classmap.txt"
        if not best_pth.is_file() or not classmap.is_file():
            return

        # Read classes
        classes = [line.strip() for line in classmap.read_text().splitlines() if line.strip()]

        # Read config path if available
        config_file = model_dir / "config.txt"
        config_path = config_file.read_text().strip() if config_file.is_file() else None

        # Checkpoint size
        size_mb = best_pth.stat().st_size / (1024 * 1024)

        models.append({
            "path": str(model_dir),
            "label": label,
            "config_path": config_path,
            "classes": classes,
            "num_classes": len(classes),
            "size_mb": round(size_mb, 1),
        })

    # Check ./model/
    top_model = root / "model"
    if top_model.is_dir():
        _scan_model_dir(top_model, "model")

    # Check exps/*/gpu*/model/ and exps/*/model/
    exps_dir = root / "exps"
    if exps_dir.is_dir():
        for model_dir in sorted(exps_dir.rglob("model")):
            if model_dir.is_dir():
                rel = model_dir.relative_to(root)
                _scan_model_dir(model_dir, str(rel))

    # Check data/*/model/
    data_dir = root / "data"
    if data_dir.is_dir():
        for model_dir in sorted(data_dir.rglob("model")):
            if model_dir.is_dir():
                rel = model_dir.relative_to(root)
                _scan_model_dir(model_dir, str(rel))

    return models


@router.post("/api/resolve-model")
async def resolve_model(body: dict):
    """Resolve a model directory into checkpoint, classmap, and config paths.

    Returns all paths needed to submit a test or infer job for a given model directory.
    """
    model_path = body.get("model_path", "").strip()
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required")

    model_dir = Path(model_path)
    if not model_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Not a directory: {model_path}")

    best_pth = model_dir / "best.pth"
    classmap = model_dir / "classmap.txt"

    if not best_pth.is_file():
        raise HTTPException(status_code=404, detail=f"best.pth not found in {model_path}")
    if not classmap.is_file():
        raise HTTPException(status_code=404, detail=f"classmap.txt not found in {model_path}")

    classes = [line.strip() for line in classmap.read_text().splitlines() if line.strip()]

    config_file = model_dir / "config.txt"
    if config_file.is_file():
        config_path = config_file.read_text().strip()
    else:
        raise HTTPException(
            status_code=422,
            detail=f"config.txt not found in {model_path}. Cannot determine model config.",
        )

    return {
        "model_path": str(model_dir),
        "checkpoint_path": str(best_pth),
        "class_map_path": str(classmap),
        "config_path": config_path,
        "classes": classes,
    }
