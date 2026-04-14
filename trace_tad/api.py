"""FastAPI pipeline API for TRACE.

Exposes the pipeline orchestrator as HTTP endpoints for future web UI integration.
Run with: trace api [--host 0.0.0.0] [--port 8001]
"""
import asyncio
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from trace_tad.pipeline import (
    run_pipeline,
    load_job_state,
    list_jobs,
    cancel_job,
    save_job_state,
    _resolve_work_dir,
    JobPhase,
)
from trace_tad.export import export_results

app = FastAPI(title="TRACE Pipeline API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PipelineSubmitRequest(BaseModel):
    config: str
    mode: str = "full"  # "full" | "train_only" | "infer_only"
    checkpoint: Optional[str] = None
    infer_videos: list[str] = []
    export_format: Optional[str] = None
    cfg_options: dict = {}
    seed: int = 42


class PipelineStatusResponse(BaseModel):
    job_id: str
    phase: str
    mode: str
    config: str
    work_dir: str
    checkpoint: Optional[str]
    train_epoch: int
    train_max_epoch: int
    infer_video_idx: int
    infer_video_total: int
    error: Optional[str]
    result_path: Optional[str]
    created_at: float
    updated_at: float


class PipelineSubmitResponse(BaseModel):
    job_id: str
    work_dir: str
    status: str = "submitted"


# In-memory registry of active job threads (job_id -> Thread)
_active_threads: dict[str, threading.Thread] = {}

# Map job_id -> work_dir for lookup without requiring work_dir in every request
_job_work_dirs: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/pipeline/submit", response_model=PipelineSubmitResponse)
def submit_pipeline(req: PipelineSubmitRequest):
    """Submit a pipeline job for background execution."""
    if req.mode == "infer_only" and not req.checkpoint:
        raise HTTPException(400, "checkpoint is required for infer_only mode")

    # Resolve work_dir upfront so we can return it
    work_dir, _ = _resolve_work_dir(req.config, req.seed)

    import uuid
    job_id = uuid.uuid4().hex[:12]

    # Track work_dir for this job
    _job_work_dirs[job_id] = work_dir

    def _run():
        try:
            run_pipeline(
                config=req.config,
                mode=req.mode,
                checkpoint=req.checkpoint,
                infer_videos=req.infer_videos,
                export_format=req.export_format,
                cfg_options=req.cfg_options,
                seed=req.seed,
                job_id=job_id,
            )
        finally:
            _active_threads.pop(job_id, None)

    t = threading.Thread(target=_run, daemon=True)
    _active_threads[job_id] = t
    t.start()

    return PipelineSubmitResponse(job_id=job_id, work_dir=work_dir)


@app.get("/api/pipeline/jobs")
def get_jobs(work_dir: Optional[str] = None):
    """List all pipeline jobs.  If work_dir not given, lists from all known jobs."""
    if work_dir:
        jobs = list_jobs(work_dir)
    else:
        # Collect from all known work_dirs
        seen = set()
        jobs = []
        for wdir in set(_job_work_dirs.values()):
            for job in list_jobs(wdir):
                if job.job_id not in seen:
                    seen.add(job.job_id)
                    jobs.append(job)
    return [_state_to_response(j) for j in jobs]


@app.get("/api/pipeline/jobs/{job_id}", response_model=PipelineStatusResponse)
def get_job_status(job_id: str, work_dir: Optional[str] = None):
    """Get the status of a specific pipeline job."""
    state = _find_job(job_id, work_dir)
    return _state_to_response(state)


@app.get("/api/pipeline/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str, work_dir: Optional[str] = None):
    """SSE endpoint for real-time job progress updates."""
    # Validate the job exists
    _find_job(job_id, work_dir)
    resolved_work_dir = work_dir or _job_work_dirs.get(job_id)

    async def event_stream():
        last_updated = 0.0
        while True:
            state = load_job_state(resolved_work_dir, job_id)
            if state is None:
                yield f"event: error\ndata: Job not found\n\n"
                break

            if state.updated_at > last_updated:
                last_updated = state.updated_at
                import json
                data = json.dumps(_state_to_dict(state))
                yield f"data: {data}\n\n"

            if state.phase in (JobPhase.COMPLETED, JobPhase.FAILED):
                break

            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/pipeline/jobs/{job_id}/results")
def get_job_results(job_id: str, work_dir: Optional[str] = None):
    """Get the results of a completed pipeline job."""
    state = _find_job(job_id, work_dir)

    if state.phase != JobPhase.COMPLETED:
        raise HTTPException(400, f"Job is not completed (phase={state.phase})")

    # Return result_detection.json content
    import json
    import os
    result_path = os.path.join(state.work_dir, "result_detection.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    raise HTTPException(404, "Result file not found")


@app.post("/api/pipeline/jobs/{job_id}/cancel")
def cancel_pipeline_job(job_id: str, work_dir: Optional[str] = None):
    """Cancel a running pipeline job."""
    resolved = work_dir or _job_work_dirs.get(job_id)
    if not resolved:
        raise HTTPException(404, "Job not found")

    ok = cancel_job(resolved, job_id)
    if not ok:
        raise HTTPException(400, "Job is not running or not found")
    return {"status": "cancelled", "job_id": job_id}


@app.get("/alive")
def health_check():
    return {"status": "ok", "service": "trace-pipeline"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_job(job_id, work_dir=None):
    resolved = work_dir or _job_work_dirs.get(job_id)
    if not resolved:
        raise HTTPException(404, f"Job {job_id} not found. Provide work_dir parameter.")
    state = load_job_state(resolved, job_id)
    if state is None:
        raise HTTPException(404, f"Job {job_id} not found in {resolved}")
    return state


def _state_to_dict(state):
    return dict(
        job_id=state.job_id,
        phase=state.phase,
        mode=state.mode,
        config=state.config,
        work_dir=state.work_dir,
        checkpoint=state.checkpoint,
        train_epoch=state.train_epoch,
        train_max_epoch=state.train_max_epoch,
        infer_video_idx=state.infer_video_idx,
        infer_video_total=state.infer_video_total,
        error=state.error,
        result_path=state.result_path,
        created_at=state.created_at,
        updated_at=state.updated_at,
    )


def _state_to_response(state):
    return PipelineStatusResponse(**_state_to_dict(state))
