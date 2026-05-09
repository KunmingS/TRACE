"""FastAPI router for TRACE training, testing, and inference jobs."""
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from trace_tad.jobs import (
    JobManager,
    JobInfo,
    JobStatus,
    JobType,
    TrainRequest,
    TrainTuneRequest,
    TestRequest,
    InferRequest,
    PrepRequest,
)
from trace_tad.model_artifacts import resolve_model_dir
from trace_tad.training_resources import estimate_training_resources
from trace_tad.pipeline_plan import (
    PipelineCommand,
    PipelineSpec,
    PipelineSpecError,
    build_pipeline_command,
)

router = APIRouter()

# Module-level singleton job manager
manager = JobManager(max_concurrency=1)


class TrainingEstimateRequest(BaseModel):
    work_dir: Optional[str] = None
    explicit_pairs: list[str] = Field(default_factory=list)
    clip_frames: int = 768
    cache_resolution: int = 144


def _require_cuda_or_400():
    """Raise HTTP 400 with a structured detail body if CUDA is unavailable.

    Imports torch lazily to keep server startup fast on hosts that never run
    GPU jobs (e.g. annotation-only deployments).
    """
    import torch
    if not torch.cuda.is_available():
        raise HTTPException(
            status_code=400,
            detail={
                "code": "CUDA_UNAVAILABLE",
                "message": "This server has no CUDA-capable GPU. Training/eval/inference require one.",
            },
        )


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
    _require_cuda_or_400()
    job = manager.start_train_job(request)
    return job


@router.post("/api/jobs/train-tune", status_code=202, response_model=JobInfo)
async def submit_train_tune_job(request: TrainTuneRequest):
    """Submit a train dataloader resource tuning job."""
    job = manager.start_train_tune_job(request)
    return job


@router.post("/api/jobs/test", status_code=202, response_model=JobInfo)
async def submit_test_job(request: TestRequest):
    """Submit a test/evaluation job to the queue."""
    _require_cuda_or_400()
    job = manager.start_test_job(request)
    return job


@router.post("/api/jobs/infer", status_code=202, response_model=JobInfo)
async def submit_infer_job(request: InferRequest):
    """Submit an inference job to the queue (no annotations needed)."""
    _require_cuda_or_400()
    job = manager.start_infer_job(request)
    return job


@router.post("/api/jobs/prep", status_code=202, response_model=JobInfo)
async def submit_prep_job(request: PrepRequest):
    """Submit a dataset preparation job that creates a model artifact folder."""
    job = manager.start_prep_job(request)
    return job


@router.post("/api/training/estimate")
async def estimate_training_job_resources(request: TrainingEstimateRequest):
    """Estimate resources for selected training pairs before prep/train."""
    try:
        return estimate_training_resources(
            work_dir=request.work_dir,
            explicit_pairs=request.explicit_pairs,
            clip_frames=request.clip_frames,
            cache_resolution=request.cache_resolution,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/pipeline/command", response_model=PipelineCommand)
async def build_pipeline_cli_command(request: PipelineSpec):
    """Build a copy-pasteable `trace pipeline ...` command for a UI pipeline."""
    try:
        return build_pipeline_command(request)
    except PipelineSpecError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/jobs", response_model=list[JobInfo])
async def list_jobs(
    type: Optional[str] = Query(None, description="Filter by job type (train/train-tune/test/infer/prep)"),
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


@router.delete("/api/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    """Delete a finished job from the registry and remove its log file."""
    try:
        manager.delete_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return None


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

    Scans common project subtrees for valid model_* directories.
    """
    root = _find_project_root()
    if root is None:
        raise HTTPException(status_code=500, detail="Cannot find project root")

    models = []
    seen_model_dirs = set()

    def _scan_model_dir(model_dir: Path, label: str):
        try:
            info = resolve_model_dir(model_dir)
        except (FileNotFoundError, ValueError):
            return
        if info["model_dir"] in seen_model_dirs:
            return
        seen_model_dirs.add(info["model_dir"])

        # Read classes
        classmap = Path(info["class_map"])
        classes = [line.strip() for line in classmap.read_text().splitlines() if line.strip()]

        # Checkpoint size
        size_mb = Path(info["checkpoint"]).stat().st_size / (1024 * 1024)

        models.append({
            "path": info["model_dir"],
            "label": label,
            "config_path": info["config_path"],
            "classes": classes,
            "num_classes": len(classes),
            "size_mb": round(size_mb, 1),
        })

    for base in (root, root / "data", root / "exps"):
        if not base.is_dir():
            continue
        for model_dir in sorted(base.rglob("model_*")):
            if model_dir.is_dir():
                rel = model_dir.relative_to(root) if model_dir.is_relative_to(root) else model_dir
                _scan_model_dir(model_dir, str(rel))

    return models


@router.post("/api/resolve-model")
async def resolve_model(body: dict):
    """Resolve a model directory into checkpoint, classmap, and config paths.

    Returns all paths needed to submit a test or infer job for a given model directory.
    """
    model_path = (body.get("model_dir") or body.get("model_path") or "").strip()
    if not model_path:
        raise HTTPException(status_code=400, detail="model_dir is required")

    model_dir = Path(model_path)
    if not model_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Not a directory: {model_path}")

    try:
        info = resolve_model_dir(model_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    classmap = Path(info["class_map"])
    classes = [line.strip() for line in classmap.read_text().splitlines() if line.strip()]

    return {
        "model_dir": info["model_dir"],
        "checkpoint_path": info["checkpoint"],
        "class_map_path": info["class_map"],
        "config_path": info["config_path"],
        "classes": classes,
    }
