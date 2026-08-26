"""Argv construction and subprocess execution for the `trace` CLI's pipeline steps.

Each `run_*` builds the command line for one step of the pipeline, runs it to
completion with its output teed to a log file, and returns the exit code. Steps
run one at a time, in the caller's process — there is no queue, no registry and
no background thread.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from vtrace.model_artifacts import (
    create_eval_dir,
    create_model_dir,
    resolve_model_dir,
)


class TrainRequest(BaseModel):
    config_path: str
    model_dir: str
    nproc: int = 1
    seed: int = 42
    resume: Optional[str] = None
    not_eval: bool = False
    disable_deterministic: bool = False
    dataset_dir: Optional[str] = None
    annotation_path: Optional[str] = None
    class_map: Optional[str] = None
    pretrained: Optional[str] = None
    cfg_options: Optional[dict] = None
    explicit_pairs: Optional[List[str]] = None


class TrainTuneProfile(BaseModel):
    name: str
    num_workers: int
    decode_threads: int
    prefetch_factor: int


class TrainTuneRequest(BaseModel):
    config_path: str
    model_dir: str
    annotation_path: str
    class_map: str
    profiles: Optional[List[TrainTuneProfile]] = None


class TestRequest(BaseModel):
    model_dir: str
    config_path: Optional[str] = None
    checkpoint: Optional[str] = None
    nproc: int = 1
    seed: int = 42
    not_eval: bool = False
    profile: bool = False
    auto_tune: bool = False
    output_dir: Optional[str] = None
    dataset_dir: Optional[str] = None
    annotation_path: Optional[str] = None
    class_map: Optional[str] = None
    cfg_options: Optional[dict] = None


class InferRequest(BaseModel):
    model_dir: str
    config_path: Optional[str] = None
    checkpoint: Optional[str] = None
    input: str
    class_map: Optional[str] = None
    output: Optional[str] = None
    output_dir: Optional[str] = None
    seed: int = 42
    profile: bool = False
    auto_tune: bool = False
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    cfg_options: Optional[dict] = None
    included_stems: Optional[List[str]] = None


class PrepRequest(BaseModel):
    work_dir: str
    model_dir: Optional[str] = None
    train_ratio: float = 0.8
    # Decode proxy built next to each source video during prep. 0 disables it
    # and decodes from the originals.
    proxy_resolution: int = 144
    proxy_aspect: bool = False
    proxy_crf: int = 23
    proxy_workers: Optional[int] = None
    included_stems: Optional[List[str]] = None
    explicit_pairs: Optional[List[str]] = None


# ── runner ──


@dataclass
class StepResult:
    """Outcome of one pipeline step."""

    returncode: int
    work_dir: str
    log_file: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _find_project_root():
    """Locate the TRACE project root by finding tools/train.py."""
    # Check relative to the package install location
    pkg_dir = Path(__file__).resolve().parent.parent
    if (pkg_dir / "tools" / "train.py").is_file():
        return str(pkg_dir)

    # Check relative to cwd
    cwd = Path.cwd()
    if (cwd / "tools" / "train.py").is_file():
        return str(cwd)

    return None


def _run(cmd: list[str], *, work_dir: str, log_file: str) -> StepResult:
    """Run one step to completion, teeing its output to `log_file` and stdout.

    Reads the child's pipe rather than tailing the log file, so output appears
    as the step produces it. Any exception on the way out (KeyboardInterrupt
    included) terminates the child first: an interrupted `vtrace train` must not
    leave a training process behind.
    """
    project_root = _find_project_root()
    if project_root is None:
        raise RuntimeError("Cannot find TRACE project root (tools/train.py not found)")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=project_root,
            text=True,
            errors="replace",
            bufsize=1,
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                fh.write(line)
            returncode = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        finally:
            proc.stdout.close()

    return StepResult(returncode=returncode, work_dir=work_dir, log_file=log_file)


# ── steps ──


def run_prep(request: PrepRequest) -> StepResult:
    """Prepare a dataset: write dataset.json / classmap.txt for the video/CSV pairs."""
    if not request.model_dir:
        request.model_dir = create_model_dir(request.work_dir)
    return _run(
        _prep_command(request),
        work_dir=request.model_dir,
        log_file=str(Path(request.model_dir) / "prep.log"),
    )


def run_train(request: TrainRequest) -> StepResult:
    """Train a model."""
    return _run(
        _train_command(request),
        work_dir=request.model_dir,
        log_file=str(Path(request.model_dir) / "job.log"),
    )


def run_train_tune(request: TrainTuneRequest) -> StepResult:
    """Benchmark dataloader settings and recommend a resource profile."""
    return _run(
        _train_tune_command(request),
        work_dir=request.model_dir,
        log_file=str(Path(request.model_dir) / "train_tune.log"),
    )


def run_test(request: TestRequest) -> StepResult:
    """Evaluate a trained model."""
    _resolve_model_request(request)
    if not request.output_dir:
        request.output_dir = create_eval_dir(request.model_dir)
    return _run(
        _test_command(request),
        work_dir=request.output_dir,
        log_file=str(Path(request.output_dir) / "job.log"),
    )


def run_infer(request: InferRequest) -> StepResult:
    """Predict behavior segments on new videos.

    The results are the `<video>.predict.csv` files the run writes beside each
    video, so nothing else needs to survive it: the engine's scratch (its log and
    the raw per-frame `result_detection.json`) goes to a temporary directory that
    is removed on success. A failed run keeps it, and `StepResult.log_file` then
    points at a log that is still there to read.
    """
    _resolve_model_request(request)
    if request.output_dir:
        # Caller chose a directory; everything stays in it, scratch included.
        return _run(
            _infer_command(request),
            work_dir=request.output_dir,
            log_file=str(Path(request.output_dir) / "job.log"),
        )

    scratch = tempfile.mkdtemp(prefix="trace-predict-")
    request.output_dir = scratch
    result = _run(
        _infer_command(request),
        work_dir=scratch,
        log_file=str(Path(scratch) / "job.log"),
    )
    if result.ok:
        shutil.rmtree(scratch, ignore_errors=True)
    return result


# ── argv construction ──


def _resolve_model_request(request):
    info = resolve_model_dir(request.model_dir)
    if not request.config_path:
        request.config_path = info["config_path"]
    if not request.checkpoint:
        request.checkpoint = info["checkpoint"]
    if not request.class_map:
        request.class_map = info["class_map"]


def _train_command(request: TrainRequest) -> list[str]:
    """Build the subprocess command for a training step."""
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


def _test_command(request: TestRequest) -> list[str]:
    """Build the subprocess command for a test/inference step."""
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


def _train_tune_command(request: TrainTuneRequest) -> list[str]:
    """Build the subprocess command for a train resource tuning step."""
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


def _infer_command(request: InferRequest) -> list[str]:
    """Build the subprocess command for an inference step."""
    cmd = [
        sys.executable, "tools/infer.py", request.config_path,
        "--checkpoint", request.checkpoint,
        "--input", request.input,
        "--class-map", request.class_map,
        "--seed", str(request.seed),
    ]
    if request.output:
        cmd.extend(["--output", request.output])
    cmd.extend(["--threshold", str(request.threshold)])
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


def _prep_command(request: PrepRequest) -> list[str]:
    """Build the subprocess command for a dataset preparation step."""
    cmd = [
        sys.executable, "tools/prep_dataset.py", request.work_dir,
        "--train-ratio", str(request.train_ratio),
        "--proxy-resolution", str(request.proxy_resolution),
        "--proxy-crf", str(request.proxy_crf),
        "--output-dir", request.model_dir,
        "--output", str(Path(request.model_dir) / "prep_result.json"),
    ]
    if request.proxy_aspect:
        cmd.append("--proxy-aspect")
    if request.proxy_workers:
        cmd.extend(["--proxy-workers", str(request.proxy_workers)])
    if request.explicit_pairs:
        cmd.extend(["--pairs", *request.explicit_pairs])
    if request.included_stems:
        cmd.extend(["--include-stems", *request.included_stems])
    return cmd
