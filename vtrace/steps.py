"""Argv construction and subprocess execution for the `trace` CLI's pipeline steps.

Each `run_*` builds the command line for one step of the pipeline, runs it to
completion with its output teed to a log file, and returns the exit code. Steps
run one at a time, in the caller's process — there is no queue, no registry and
no background thread.
"""
import json
import os
import re
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
    seed: int = 42
    resume: Optional[str] = None
    not_eval: bool = False
    disable_deterministic: bool = False
    dataset_dir: Optional[str] = None
    annotation_path: Optional[str] = None
    class_map: Optional[str] = None
    # Evaluation data, when the run has any: a separate corpus with its own
    # dataset.json. Unset means training only — checkpoints, no best.pth.
    eval_annotation_path: Optional[str] = None
    eval_data_dir: Optional[str] = None
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
    # Pick the cutoff per class against the annotation CSV beside each video,
    # when there is one. Only the demo asks for this: it tunes on the video it
    # reports, which is a demonstration, not an evaluation.
    tune_threshold: bool = False
    # Also write `<stem>.pred.scores.npz` beside each prediction file — every
    # frame's score for every class — so frame-level curves and AP can be drawn
    # after the fact. The demo asks for it to plot precision-recall over its
    # test videos; a plain `vtrace predict` writes only the CSVs.
    frame_scores: bool = False
    cfg_options: Optional[dict] = None
    included_stems: Optional[List[str]] = None


class PrepRequest(BaseModel):
    work_dir: str
    model_dir: Optional[str] = None
    # `train` or `validation`. Nothing is split during prep; a corpus is one or
    # the other, and the caller knows which because it chose the folder.
    subset: str = "train"
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


# Colour belongs on the terminal, not in a file someone greps a year later.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# A redraw of a multi-line block starts by moving the cursor back over it.
_CURSOR_UP = re.compile(r"\x1b\[(\d*)A")


class _LogTee:
    """Turns a step's terminal output into something worth keeping in a file.

    Two things a terminal does are noise on disk. A progress bar redraws one
    line with `\r`, so only the last frame of each line is kept. A live plot
    redraws a whole block by moving the cursor up over it, so those lines are
    dropped outright: on seeing `ESC[nA`, the last n lines held back are the
    ones being overwritten and go away, and everything older is now settled and
    can be written out.

    Holding back only the current block is what keeps this exact — nothing is
    guessed from timing, and `tail -f` lags by one frame at most.
    """

    def __init__(self, handle):
        self.handle = handle
        self.held = []
        self.pending = b""

    def feed(self, chunk: bytes) -> None:
        self.pending += chunk
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            self._line(line.decode("utf-8", errors="replace"))

    def _line(self, text: str) -> None:
        match = _CURSOR_UP.match(text)
        if match:
            rewound = int(match.group(1) or 1)
            if rewound:
                del self.held[max(0, len(self.held) - rewound):]
            self._settle()
            text = text[match.end():]
        # A bar's frames are separated by `\r`; the last one is its final state.
        self.held.append(_ANSI.sub("", text.rsplit("\r", 1)[-1]).rstrip())

    def _settle(self) -> None:
        for line in self.held:
            self.handle.write(line + "\n")
        self.held.clear()
        self.handle.flush()

    def close(self) -> None:
        if self.pending.strip():
            self._line(self.pending.decode("utf-8", errors="replace"))
        self.pending = b""
        self._settle()


def _child_env() -> dict:
    """The child's environment, told what kind of terminal it is writing to.

    A step runs with its stdout on a pipe, so Rich and friends would correctly
    conclude there is nobody to colour for — but this process is teeing that
    pipe straight to a real terminal. FORCE_COLOR passes on what the child
    cannot see, and COLUMNS passes on how wide it is.
    """
    env = dict(os.environ)
    if sys.stdout.isatty() and not env.get("NO_COLOR"):
        env["FORCE_COLOR"] = "1"
        env["COLUMNS"] = str(shutil.get_terminal_size((100, 24)).columns)
    # decord 0.6 occasionally gives up on a frame near EOF with "Unable to
    # handle EOF because it takes too long to retrieve last few frames" (its
    # default budget is 10240 retries). Give it more room; VideoDecode also
    # reopens the reader and retries, so this is belt and braces.
    env.setdefault("DECORD_EOF_RETRY_MAX", "40960")
    return env


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
        # Binary, not text: a text-mode pipe is opened with universal newlines,
        # which translates the bare `\r` a progress bar redraws with into `\n`.
        # That is what turned one updating line into a hundred printed ones.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=project_root,
            env=_child_env(),
        )
        tee = _LogTee(fh)
        try:
            while True:
                # read1 returns what has arrived instead of waiting to fill a
                # buffer, so the bar moves while the step runs.
                chunk = proc.stdout.read1(65536)
                if not chunk:
                    break
                # Straight through, escapes intact: the terminal is what makes
                # them redraw in place. The log is scrollback, not a live
                # display, and gets the settled version.
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                tee.feed(chunk)
            tee.close()
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

    The results are the `<video>.pred.csv` files the run writes beside each
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
    if request.eval_annotation_path:
        cfg_opts.append(f"eval_annotation_path={request.eval_annotation_path}")
    if request.eval_data_dir:
        cfg_opts.append(f"eval_data_path={request.eval_data_dir}")
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
    if request.tune_threshold:
        cmd.append("--tune-threshold")
    if request.frame_scores:
        cmd.append("--frame-scores")
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
        "--subset", request.subset,
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
