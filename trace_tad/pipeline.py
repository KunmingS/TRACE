"""Pipeline orchestrator for TRACE.

Chains training -> checkpoint discovery -> multi-video inference -> result export
into a single automated workflow.  Designed to be called from the CLI (blocking)
or from the API (in a background thread).
"""
import json
import os
import re
import subprocess
import sys
import signal
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

from trace_tad.export import export_results


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

class JobPhase(str, Enum):
    PENDING = "pending"
    TRAINING = "training"
    INFERENCE = "inference"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobState:
    job_id: str
    config: str
    work_dir: str
    phase: str = JobPhase.PENDING
    mode: str = "full"  # "full" | "train_only" | "infer_only"
    checkpoint: Optional[str] = None
    infer_videos: list = field(default_factory=list)
    export_format: Optional[str] = None
    cfg_options: dict = field(default_factory=dict)
    seed: int = 42
    # progress
    train_epoch: int = 0
    train_max_epoch: int = 0
    infer_video_idx: int = 0
    infer_video_total: int = 0
    error: Optional[str] = None
    result_path: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    pid: Optional[int] = None


def _job_dir(work_dir, job_id):
    return os.path.join(work_dir, ".pipeline", job_id)


def save_job_state(state):
    """Atomically persist job state to disk."""
    d = _job_dir(state.work_dir, state.job_id)
    os.makedirs(d, exist_ok=True)
    state.updated_at = time.time()
    tmp = os.path.join(d, "job.json.tmp")
    target = os.path.join(d, "job.json")
    with open(tmp, "w") as f:
        json.dump(asdict(state), f, indent=2)
    os.replace(tmp, target)


def load_job_state(work_dir, job_id):
    """Load job state from disk.  Returns None if not found."""
    path = os.path.join(_job_dir(work_dir, job_id), "job.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return JobState(**data)


def list_jobs(work_dir):
    """List all pipeline jobs under a work_dir."""
    pipeline_dir = os.path.join(work_dir, ".pipeline")
    if not os.path.isdir(pipeline_dir):
        return []
    jobs = []
    for name in sorted(os.listdir(pipeline_dir)):
        state = load_job_state(work_dir, name)
        if state is not None:
            jobs.append(state)
    return jobs


def cancel_job(work_dir, job_id):
    """Cancel a running job by sending SIGTERM to its subprocess."""
    state = load_job_state(work_dir, job_id)
    if state is None:
        return False
    if state.pid is not None and state.phase in (JobPhase.TRAINING, JobPhase.INFERENCE):
        try:
            os.kill(state.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        state.phase = JobPhase.FAILED
        state.error = "Cancelled by user"
        save_job_state(state)
        return True
    return False


# ---------------------------------------------------------------------------
# Subprocess runner with progress parsing
# ---------------------------------------------------------------------------

_EPOCH_RE = re.compile(r"Epoch\s*\[(\d+)/(\d+)\]")
_EVAL_RE = re.compile(r"Testing Starts")


def _run_subprocess(cmd, state, phase, log_path, on_progress=None):
    """Run a command, stream stdout to a log file, and parse progress."""
    state.phase = phase
    save_job_state(state)

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        state.pid = proc.pid
        save_job_state(state)

        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()

            # Parse training progress
            if phase == JobPhase.TRAINING:
                m = _EPOCH_RE.search(line)
                if m:
                    state.train_epoch = int(m.group(1))
                    state.train_max_epoch = int(m.group(2))
                    save_job_state(state)

            if on_progress:
                on_progress(line, state)

        proc.wait()
        state.pid = None
        save_job_state(state)

    if proc.returncode != 0:
        state.phase = JobPhase.FAILED
        state.error = f"Subprocess exited with code {proc.returncode}"
        save_job_state(state)
        return False

    return True


# ---------------------------------------------------------------------------
# Synthetic annotation for arbitrary video inference
# ---------------------------------------------------------------------------

def _create_infer_annotation(videos, job_dir):
    """Create a minimal annotation JSON for inference on arbitrary videos.

    Returns:
        (annotation_path, data_path) tuple.
    """
    database = {}
    # All videos must be in the same directory (or we pick the first one's dir)
    for video_path in videos:
        abs_path = os.path.abspath(video_path)
        name = os.path.splitext(os.path.basename(abs_path))[0]
        database[name] = {
            "subset": "testing",
            "duration": -1,
            "annotations": [],
        }

    ann = {"version": "TRACE-INFER", "database": database}
    ann_path = os.path.join(job_dir, "infer_annotation.json")
    with open(ann_path, "w") as f:
        json.dump(ann, f, indent=2)

    # Use the directory of the first video as data_path
    data_path = os.path.dirname(os.path.abspath(videos[0]))
    return ann_path, data_path


# ---------------------------------------------------------------------------
# Config resolution helpers
# ---------------------------------------------------------------------------

def _resolve_work_dir(config_path, seed=42, exp_id=0):
    """Resolve work_dir from a config file without importing the full model stack."""
    from trace_tad.config import Config
    cfg = Config.fromfile(config_path)
    # Replicate update_workdir logic
    work_dir = os.path.join(cfg.work_dir, f"gpu1_id{exp_id}/")
    return work_dir, cfg


def _resolve_checkpoint(work_dir, checkpoint=None):
    """Find the best checkpoint, falling back to explicit path."""
    if checkpoint:
        return checkpoint
    best = os.path.join(work_dir, "checkpoint", "best.pth")
    if os.path.exists(best):
        return best
    return None


def _build_cfg_options_args(cfg_options):
    """Convert a dict of config overrides to --cfg-options CLI args."""
    if not cfg_options:
        return []
    parts = ["--cfg-options"]
    for k, v in cfg_options.items():
        parts.append(f"{k}={v}")
    return parts


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    config,
    mode="full",
    checkpoint=None,
    infer_videos=None,
    export_format=None,
    cfg_options=None,
    seed=42,
    job_id=None,
    on_progress=None,
):
    """Run the full TRACE pipeline: train -> infer -> export.

    Args:
        config: Path to config file.
        mode: "full", "train_only", or "infer_only".
        checkpoint: Explicit checkpoint path (required for infer_only).
        infer_videos: List of video paths for inference (optional).
        export_format: "csv" or None.
        cfg_options: Dict of config overrides.
        seed: Random seed.
        job_id: Unique job identifier (auto-generated if None).
        on_progress: Optional callback(line, state) for each stdout line.

    Returns:
        JobState with final status.
    """
    config = os.path.abspath(config)
    job_id = job_id or uuid.uuid4().hex[:12]
    cfg_options = cfg_options or {}
    infer_videos = infer_videos or []

    # Resolve work_dir from config
    work_dir, _ = _resolve_work_dir(config, seed)

    # Initialize job state
    state = JobState(
        job_id=job_id,
        config=config,
        work_dir=work_dir,
        mode=mode,
        checkpoint=checkpoint,
        infer_videos=infer_videos,
        export_format=export_format,
        cfg_options=cfg_options,
        seed=seed,
        infer_video_total=len(infer_videos),
        created_at=time.time(),
    )

    job_dir = _job_dir(work_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    save_job_state(state)

    print(f"Pipeline job {job_id} started (mode={mode})")
    print(f"  Config:   {config}")
    print(f"  Work dir: {work_dir}")
    print(f"  Job dir:  {job_dir}")

    # ---- TRAINING ----
    if mode in ("full", "train_only"):
        train_cmd = [
            sys.executable, "-m", "tools.train",
            config,
            "--seed", str(seed),
        ] + _build_cfg_options_args(cfg_options)

        train_log = os.path.join(job_dir, "train.log")
        print(f"\n--- Training ---")
        ok = _run_subprocess(train_cmd, state, JobPhase.TRAINING, train_log, on_progress)
        if not ok:
            print(f"Training failed: {state.error}")
            print(f"  See log: {train_log}")
            return state
        print(f"Training complete. Log: {train_log}")

    # ---- CHECKPOINT RESOLUTION ----
    if mode in ("full", "infer_only"):
        ckpt = _resolve_checkpoint(work_dir, checkpoint)
        if ckpt is None:
            state.phase = JobPhase.FAILED
            state.error = "No checkpoint found. Train first or provide --checkpoint."
            save_job_state(state)
            print(f"Error: {state.error}")
            return state
        state.checkpoint = ckpt
        save_job_state(state)
        print(f"\nUsing checkpoint: {ckpt}")

    # ---- INFERENCE ----
    if mode in ("full", "infer_only"):
        infer_cfg_options = dict(cfg_options)

        # If custom videos provided, create synthetic annotation
        if infer_videos:
            ann_path, data_path = _create_infer_annotation(infer_videos, job_dir)
            infer_cfg_options["dataset.test.ann_file"] = ann_path
            infer_cfg_options["dataset.test.data_path"] = data_path
            state.infer_video_total = len(infer_videos)
            print(f"Inference on {len(infer_videos)} video(s)")
        else:
            print("Inference on dataset test split")

        infer_cmd = [
            sys.executable, "-m", "tools.test",
            config,
            "--checkpoint", state.checkpoint,
            "--seed", str(seed),
            "--auto-tune",
        ] + _build_cfg_options_args(infer_cfg_options)

        infer_log = os.path.join(job_dir, "infer.log")
        print(f"\n--- Inference ---")
        ok = _run_subprocess(infer_cmd, state, JobPhase.INFERENCE, infer_log, on_progress)
        if not ok:
            print(f"Inference failed: {state.error}")
            print(f"  See log: {infer_log}")
            return state
        print(f"Inference complete. Log: {infer_log}")

    # ---- EXPORT ----
    if export_format:
        state.phase = JobPhase.EXPORTING
        save_job_state(state)

        result_json = os.path.join(work_dir, "result_detection.json")
        if os.path.exists(result_json):
            try:
                out_path = export_results(result_json, job_dir, fmt=export_format)
                state.result_path = out_path
                print(f"\nExported results: {out_path}")
            except Exception as e:
                state.phase = JobPhase.FAILED
                state.error = f"Export failed: {e}"
                save_job_state(state)
                return state
        else:
            print(f"Warning: {result_json} not found, skipping export")

    # ---- DONE ----
    state.phase = JobPhase.COMPLETED
    save_job_state(state)
    print(f"\nPipeline job {job_id} completed successfully.")
    return state
