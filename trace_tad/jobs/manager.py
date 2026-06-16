"""Job manager for TRACE training and testing jobs.

Handles job queuing, subprocess execution, log capture, and file-based persistence.
"""
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

from .models import (
    JobInfo,
    JobStatus,
    JobType,
    TrainRequest,
    TrainTuneRequest,
    TestRequest,
    InferRequest,
    PrepRequest,
)
from trace_tad.model_artifacts import (
    create_eval_dir,
    create_model_dir,
    create_predict_dir_for_input,
    resolve_model_dir,
)

JOB_REQUEST_METADATA = {"run_id", "run_steps"}
LEGACY_MODEL_RUN_TYPES = {JobType.PREP.value, JobType.TRAIN.value}


def _find_project_root():
    """Locate the TRACE project root by finding tools/train.py."""
    # Check relative to the package install location
    pkg_dir = Path(__file__).resolve().parent.parent.parent
    if (pkg_dir / "tools" / "train.py").is_file():
        return str(pkg_dir)

    # Check relative to cwd
    cwd = Path.cwd()
    if (cwd / "tools" / "train.py").is_file():
        return str(cwd)

    return None


def _trace_home():
    """Return the TRACE home directory (~/.trace), creating it if needed."""
    home = Path.home() / ".trace"
    home.mkdir(parents=True, exist_ok=True)
    return home


def _logs_dir():
    """Return the logs directory (~/.trace/logs), creating it if needed."""
    d = _trace_home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jobs_file():
    """Return the path to the jobs persistence file."""
    return _trace_home() / "jobs.json"


class JobManager:
    """Manages training and testing jobs with a FIFO queue.

    Args:
        max_concurrency: Maximum number of jobs that can run simultaneously.
    """

    def __init__(self, max_concurrency: int = 1):
        self.max_concurrency = max_concurrency
        self._lock = threading.Lock()
        self._jobs: dict[str, JobInfo] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._queue: deque[str] = deque()  # job_ids waiting to run
        self._project_root = _find_project_root()

        # Load persisted jobs
        self._load_jobs()

    # ── Public API ──────────────────────────────────────────────────

    def start_train_job(self, request: TrainRequest) -> JobInfo:
        """Submit a training job to the queue."""
        cmd = self._build_train_command(request)
        args = request.model_dump(exclude_none=True, exclude=JOB_REQUEST_METADATA)
        log_file = str(Path(request.model_dir) / "job.log")
        return self._enqueue_job(
            JobType.TRAIN,
            request.config_path,
            cmd,
            args,
            work_dir=request.model_dir,
            log_file=log_file,
            run_id=request.run_id,
            stage="train",
            run_steps=request.run_steps,
        )

    def start_train_tune_job(self, request: TrainTuneRequest) -> JobInfo:
        """Submit a train resource tuning job to the queue."""
        cmd = self._build_train_tune_command(request)
        args = request.model_dump(exclude_none=True, exclude=JOB_REQUEST_METADATA)
        log_file = str(Path(request.model_dir) / "train_tune.log")
        return self._enqueue_job(
            JobType.TRAIN_TUNE,
            request.config_path,
            cmd,
            args,
            work_dir=request.model_dir,
            log_file=log_file,
            run_id=request.run_id,
            stage="train-tune",
            run_steps=request.run_steps,
        )

    def start_test_job(self, request: TestRequest) -> JobInfo:
        """Submit a test/evaluation job to the queue."""
        self._resolve_model_request(request)
        if not request.output_dir:
            request.output_dir = create_eval_dir(request.model_dir)
        cmd = self._build_test_command(request)
        args = request.model_dump(exclude_none=True, exclude=JOB_REQUEST_METADATA)
        log_file = str(Path(request.output_dir) / "job.log")
        return self._enqueue_job(
            JobType.TEST,
            request.config_path or request.model_dir,
            cmd,
            args,
            work_dir=request.output_dir,
            log_file=log_file,
            run_id=request.run_id,
            stage="test",
            run_steps=request.run_steps,
        )

    def start_infer_job(self, request: InferRequest) -> JobInfo:
        """Submit an inference job to the queue."""
        self._resolve_model_request(request)
        if not request.output_dir:
            request.output_dir = create_predict_dir_for_input(request.input)
        cmd = self._build_infer_command(request)
        args = request.model_dump(exclude_none=True, exclude=JOB_REQUEST_METADATA)
        log_file = str(Path(request.output_dir) / "job.log")
        return self._enqueue_job(
            JobType.INFER,
            request.config_path or request.model_dir,
            cmd,
            args,
            work_dir=request.output_dir,
            log_file=log_file,
            run_id=request.run_id,
            stage="infer",
            run_steps=request.run_steps,
        )

    def start_prep_job(self, request: PrepRequest) -> JobInfo:
        """Submit a dataset preparation job to the queue."""
        if not request.model_dir:
            request.model_dir = create_model_dir(request.work_dir)
        cmd = self._build_prep_command(request)
        args = request.model_dump(exclude_none=True, exclude=JOB_REQUEST_METADATA)
        log_file = str(Path(request.model_dir) / "prep.log")
        return self._enqueue_job(
            JobType.PREP,
            request.work_dir,
            cmd,
            args,
            work_dir=request.model_dir,
            log_file=log_file,
            run_id=request.run_id,
            stage="prep",
            run_steps=request.run_steps,
        )

    def cancel_job(self, job_id: str) -> JobInfo:
        """Cancel a queued or running job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Job {job_id} not found")

            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.finished_at = _now()
                if job_id in self._queue:
                    self._queue.remove(job_id)
            elif job.status == JobStatus.RUNNING:
                proc = self._processes.get(job_id)
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                job.status = JobStatus.CANCELLED
                job.finished_at = _now()
                self._processes.pop(job_id, None)
            else:
                raise ValueError(f"Job {job_id} is {job.status}, cannot cancel")

            self._persist()
        # Try to start next queued job
        self._try_start_next()
        return job

    def delete_job(self, job_id: str) -> None:
        """Remove a finished job from the registry and delete its log file.

        Refuses to delete jobs that are still queued or running — caller must
        cancel them first.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Job {job_id} not found")
            if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                raise ValueError(
                    f"Job {job_id} is {job.status}; cancel before deleting"
                )

            log_path = Path(job.log_file) if job.log_file else None
            self._jobs.pop(job_id, None)
            self._processes.pop(job_id, None)
            try:
                self._queue.remove(job_id)
            except ValueError:
                pass
            self._persist()

        if log_path and log_path.is_file():
            try:
                log_path.unlink()
            except OSError:
                pass

    def get_job(self, job_id: str) -> JobInfo:
        """Get info for a single job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Job {job_id} not found")
            return job.model_copy()

    def list_jobs(
        self,
        type_filter: Optional[JobType] = None,
        status_filter: Optional[JobStatus] = None,
    ) -> list[JobInfo]:
        """List jobs, optionally filtered by type and/or status."""
        with self._lock:
            jobs = list(self._jobs.values())
        if type_filter:
            jobs = [j for j in jobs if j.job_type == type_filter]
        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]
        # Most recent first
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def get_log_tail(self, job_id: str, n_lines: int = 100) -> list[str]:
        """Return the last N lines of a job's log file."""
        job = self.get_job(job_id)
        log_path = Path(job.log_file)
        if not log_path.is_file():
            return []
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [line.rstrip("\n") for line in lines[-n_lines:]]

    def stream_log(self, job_id: str) -> Generator[str, None, None]:
        """Generator that tails a job's log file, yielding new lines.

        Exits when the job finishes and all lines have been consumed.
        """
        job = self.get_job(job_id)
        log_path = Path(job.log_file)

        # Wait for log file to appear
        for _ in range(50):  # up to 5 seconds
            if log_path.is_file():
                break
            time.sleep(0.1)

        if not log_path.is_file():
            yield "[Log file not found]"
            return

        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                line = f.readline()
                if line:
                    yield line.rstrip("\n")
                else:
                    # Check if the job is still running
                    with self._lock:
                        current_job = self._jobs.get(job_id)
                    if current_job is None or current_job.status not in (
                        JobStatus.QUEUED,
                        JobStatus.RUNNING,
                    ):
                        # Read any remaining lines
                        remaining = f.read()
                        if remaining:
                            for rem_line in remaining.splitlines():
                                yield rem_line
                        return
                    time.sleep(0.2)

    def wait_for_job(self, job_id: str, stream_to=None) -> JobInfo:
        """Block until job completes. Optionally stream logs to a file-like object.

        Args:
            job_id: The job to wait for.
            stream_to: If provided, write log lines to this object (e.g. sys.stdout).
        """
        if stream_to is not None:
            for line in self.stream_log(job_id):
                stream_to.write(line + "\n")
                stream_to.flush()
        else:
            while True:
                job = self.get_job(job_id)
                if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
                    break
                time.sleep(0.5)

        return self.get_job(job_id)

    def cleanup_all(self):
        """Terminate all running jobs. Called on server shutdown."""
        with self._lock:
            for job_id, proc in list(self._processes.items()):
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                job = self._jobs.get(job_id)
                if job and job.status == JobStatus.RUNNING:
                    job.status = JobStatus.CANCELLED
                    job.finished_at = _now()
            self._processes.clear()
            self._persist()

    # ── Private Methods ─────────────────────────────────────────────

    def _enqueue_job(
        self,
        job_type: JobType,
        config_path: str,
        cmd: list[str],
        args: dict,
        *,
        work_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        run_id: Optional[str] = None,
        stage: Optional[str] = None,
        run_steps: Optional[list[str]] = None,
    ) -> JobInfo:
        """Create a job and add it to the queue."""
        job_id = uuid.uuid4().hex[:12]
        log_file = log_file or str(_logs_dir() / f"{job_id}.log")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        run_id = run_id or job_id

        job = JobInfo(
            job_id=job_id,
            job_type=job_type,
            run_id=run_id,
            stage=stage or job_type.value,
            run_steps=run_steps,
            status=JobStatus.QUEUED,
            config_path=config_path,
            created_at=_now(),
            log_file=log_file,
            work_dir=work_dir,
            args=args,
        )

        with self._lock:
            self._jobs[job_id] = job
            self._queue.append(job_id)
            # Store the command in args for subprocess launching
            job.args["_cmd"] = cmd
            self._persist()

        self._try_start_next()
        return job

    def _try_start_next(self):
        """Start the next queued job if we're below max concurrency."""
        with self._lock:
            running_count = sum(
                1 for j in self._jobs.values() if j.status == JobStatus.RUNNING
            )
            while self._queue and running_count < self.max_concurrency:
                job_id = self._queue.popleft()
                job = self._jobs.get(job_id)
                if job is None or job.status != JobStatus.QUEUED:
                    continue
                self._start_job(job)
                running_count += 1

    def _start_job(self, job: JobInfo):
        """Launch the subprocess for a job. Must hold self._lock."""
        cmd = job.args.pop("_cmd", None)
        if cmd is None:
            job.status = JobStatus.FAILED
            job.error_message = "No command found for job"
            job.finished_at = _now()
            self._persist()
            return

        if self._project_root is None:
            job.status = JobStatus.FAILED
            job.error_message = "Cannot find TRACE project root (tools/train.py not found)"
            job.finished_at = _now()
            self._persist()
            return

        try:
            log_fh = open(job.log_file, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=self._project_root,
            )
            job.status = JobStatus.RUNNING
            job.started_at = _now()
            job.pid = proc.pid
            if not job.work_dir:
                job.work_dir = str(self._project_root) if self._project_root else None
            self._processes[job.job_id] = proc
            self._persist()

            # Monitor in a daemon thread
            t = threading.Thread(
                target=self._monitor_job,
                args=(job.job_id, proc, log_fh),
                daemon=True,
            )
            t.start()
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.finished_at = _now()
            self._persist()

    def _monitor_job(self, job_id: str, proc: subprocess.Popen, log_fh):
        """Wait for a subprocess to finish and update job status."""
        try:
            proc.wait()
        finally:
            log_fh.close()

        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == JobStatus.RUNNING:
                job.return_code = proc.returncode
                job.finished_at = _now()
                if proc.returncode == 0:
                    job.status = JobStatus.COMPLETED
                else:
                    job.status = JobStatus.FAILED
                    job.error_message = f"Process exited with code {proc.returncode}"
            self._processes.pop(job_id, None)
            self._persist()

        self._try_start_next()

    def _resolve_model_request(self, request):
        info = resolve_model_dir(request.model_dir)
        if not request.config_path:
            request.config_path = info["config_path"]
        if not request.checkpoint:
            request.checkpoint = info["checkpoint"]
        if not request.class_map:
            request.class_map = info["class_map"]

    def _build_train_command(self, request: TrainRequest) -> list[str]:
        """Build the subprocess command for a training job."""
        if request.nproc > 1:
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node", str(request.nproc),
                "tools/train.py", request.config_path,
            ]
        else:
            cmd = [sys.executable, "tools/train.py", request.config_path]

        cmd.extend(["--seed", str(request.seed)])
        if request.resume:
            cmd.extend(["--resume", request.resume])
        if request.not_eval:
            cmd.append("--not_eval")
        if request.disable_deterministic:
            cmd.append("--disable_deterministic")

        # Build --cfg-options from user-friendly flags + raw cfg_options
        cfg_opts = [f"work_dir={request.model_dir}"]
        if request.dataset_dir:
            cfg_opts.append(f"data_path={request.dataset_dir}")
        if request.annotation_path:
            cfg_opts.append(f"annotation_path={request.annotation_path}")
        if request.class_map:
            cfg_opts.append(f"class_map={request.class_map}")
        if request.pretrained:
            cfg_opts.append(f"model.projection.custom.pretrain={request.pretrained}")
        if request.cfg_options:
            for key, val in request.cfg_options.items():
                cfg_opts.append(f"{key}={val}")
        if cfg_opts:
            cmd.extend(["--cfg-options"] + cfg_opts)

        return cmd

    def _build_test_command(self, request: TestRequest) -> list[str]:
        """Build the subprocess command for a test/inference job."""
        if request.nproc > 1:
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node", str(request.nproc),
                "tools/test.py", request.config_path,
            ]
        else:
            cmd = [sys.executable, "tools/test.py", request.config_path]

        cmd.extend(["--checkpoint", request.checkpoint])
        cmd.extend(["--seed", str(request.seed)])
        if request.not_eval:
            cmd.append("--not_eval")
        if request.profile:
            cmd.append("--profile")
        if request.auto_tune:
            cmd.append("--auto-tune")

        # Build --cfg-options from user-friendly flags + raw cfg_options
        cfg_opts = [f"work_dir={request.output_dir}"]
        if request.dataset_dir:
            cfg_opts.append(f"data_path={request.dataset_dir}")
        if request.annotation_path:
            cfg_opts.append(f"annotation_path={request.annotation_path}")
        if request.class_map:
            cfg_opts.append(f"class_map={request.class_map}")
        if request.cfg_options:
            for key, val in request.cfg_options.items():
                cfg_opts.append(f"{key}={val}")
        if cfg_opts:
            cmd.extend(["--cfg-options"] + cfg_opts)

        return cmd

    def _build_train_tune_command(self, request: TrainTuneRequest) -> list[str]:
        """Build the subprocess command for a train resource tuning job."""
        cmd = [
            sys.executable,
            "tools/tune_train.py",
            request.config_path,
            "--model-dir",
            request.model_dir,
            "--annotation-path",
            request.annotation_path,
            "--class-map",
            request.class_map,
            "--output",
            str(Path(request.model_dir) / "train_tune_result.json"),
        ]
        if request.profiles:
            cmd.extend(["--profiles-json", json.dumps([p.model_dump() for p in request.profiles])])
        return cmd

    def _build_infer_command(self, request: InferRequest) -> list[str]:
        """Build the subprocess command for an inference job."""
        cmd = [
            sys.executable, "tools/infer.py", request.config_path,
            "--checkpoint", request.checkpoint,
            "--input", request.input,
            "--class-map", request.class_map,
            "--seed", str(request.seed),
        ]
        if request.output:
            cmd.extend(["--output", request.output])
        if request.threshold is not None:
            cmd.extend(["--threshold", str(request.threshold)])
        if request.annotated_video:
            cmd.append("--annotated-video")
        if request.profile:
            cmd.append("--profile")
        if request.auto_tune:
            cmd.append("--auto-tune")
        if request.included_stems:
            cmd.extend(["--include-stems", *request.included_stems])
        if request.cfg_options:
            cfg_opts = [f"{k}={v}" for k, v in request.cfg_options.items()]
        else:
            cfg_opts = []
        cfg_opts.insert(0, f"work_dir={request.output_dir}")
        if cfg_opts:
            cmd.extend(["--cfg-options"] + cfg_opts)
        return cmd

    def _build_prep_command(self, request: PrepRequest) -> list[str]:
        """Build the subprocess command for a dataset preparation job."""
        cmd = [
            sys.executable, "tools/prep_dataset.py", request.work_dir,
            "--clip-frames", str(request.clip_frames),
            "--train-ratio", str(request.train_ratio),
            "--cache-mode", request.cache_mode,
            *(["--val-ratio", str(request.val_ratio)] if request.val_ratio is not None else []),
            *(["--test-ratio", str(request.test_ratio)] if request.test_ratio is not None else []),
            "--cache-resolution", str(request.cache_resolution),
            "--cache-crf", str(request.cache_crf),
            "--output-dir", request.model_dir,
            "--output", str(Path(request.model_dir) / "prep_result.json"),
        ]
        if request.cache_workers:
            cmd.extend(["--cache-workers", str(request.cache_workers)])
        if request.explicit_pairs:
            cmd.extend(["--pairs", *request.explicit_pairs])
        if request.included_stems:
            cmd.extend(["--include-stems", *request.included_stems])
        return cmd

    # ── Persistence ─────────────────────────────────────────────────

    def _persist(self):
        """Save all jobs to disk. Must hold self._lock (or be called from __init__)."""
        data = {}
        for job_id, job in self._jobs.items():
            d = job.model_dump()
            # Don't persist internal _cmd
            d.get("args", {}).pop("_cmd", None)
            data[job_id] = d
        try:
            with open(_jobs_file(), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass  # Best-effort persistence

    def _load_jobs(self):
        """Load jobs from disk on startup."""
        path = _jobs_file()
        if not path.is_file():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for job_id, d in data.items():
                # Mark previously-running jobs as failed (stale from crash)
                if d.get("status") == JobStatus.RUNNING:
                    d["status"] = JobStatus.FAILED
                    d["error_message"] = "Server restarted while job was running"
                    d["finished_at"] = _now()
                if d.get("status") == JobStatus.QUEUED:
                    d["status"] = JobStatus.CANCELLED
                    d["error_message"] = "Server restarted while job was queued"
                    d["finished_at"] = _now()
                if not d.get("stage"):
                    d["stage"] = d.get("job_type")
                if not d.get("run_id"):
                    legacy_model_dir = _legacy_model_dir(d)
                    d["run_id"] = f"legacy:{legacy_model_dir}" if legacy_model_dir else job_id
                if not d.get("run_steps"):
                    d["run_steps"] = _legacy_run_steps(d)
                self._jobs[job_id] = JobInfo(**d)
        except (json.JSONDecodeError, OSError):
            pass  # Corrupted file, start fresh


def _now() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _legacy_model_dir(job_data: dict) -> Optional[str]:
    """Infer old prep/train run identity from persisted jobs without run_id."""
    if job_data.get("job_type") not in LEGACY_MODEL_RUN_TYPES:
        return None
    args = job_data.get("args") or {}
    if isinstance(args, dict):
        model_dir = args.get("model_dir")
        if isinstance(model_dir, str) and model_dir:
            return model_dir
    work_dir = job_data.get("work_dir")
    if isinstance(work_dir, str) and work_dir:
        return work_dir
    return None


def _legacy_run_steps(job_data: dict) -> Optional[list[str]]:
    stage = job_data.get("stage") or job_data.get("job_type")
    if stage == "prep":
        return ["train"]
    if stage in {"train", "test", "infer"}:
        return [stage]
    return None
