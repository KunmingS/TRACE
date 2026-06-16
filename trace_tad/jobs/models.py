"""Pydantic models for TRACE job management."""
from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class JobType(str, Enum):
    TRAIN = "train"
    TEST = "test"
    INFER = "infer"
    PREP = "prep"
    TRAIN_TUNE = "train-tune"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobInfo(BaseModel):
    job_id: str
    job_type: JobType
    run_id: Optional[str] = None
    stage: Optional[str] = None
    run_steps: Optional[List[str]] = None
    status: JobStatus
    config_path: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    pid: Optional[int] = None
    return_code: Optional[int] = None
    log_file: str
    work_dir: Optional[str] = None
    error_message: Optional[str] = None
    args: dict = Field(default_factory=dict)


class TrainRequest(BaseModel):
    run_id: Optional[str] = None
    run_steps: Optional[List[str]] = None
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
    run_id: Optional[str] = None
    run_steps: Optional[List[str]] = None
    config_path: str
    model_dir: str
    annotation_path: str
    class_map: str
    profiles: Optional[List[TrainTuneProfile]] = None


class TestRequest(BaseModel):
    run_id: Optional[str] = None
    run_steps: Optional[List[str]] = None
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
    run_id: Optional[str] = None
    run_steps: Optional[List[str]] = None
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
    annotated_video: bool = False
    # None ⇒ unspecified: infer.py auto-applies the recommended thresholds saved
    # at training time (recommended_thresholds.json), falling back to 0.0.
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cfg_options: Optional[dict] = None
    included_stems: Optional[List[str]] = None


class PrepRequest(BaseModel):
    run_id: Optional[str] = None
    run_steps: Optional[List[str]] = None
    work_dir: str
    model_dir: Optional[str] = None
    clip_frames: int = 768
    train_ratio: float = 0.7
    val_ratio: Optional[float] = None
    test_ratio: Optional[float] = None
    cache_mode: Literal["virtual", "cached_video"] = "cached_video"
    cache_resolution: int = 144
    cache_crf: int = 23
    cache_workers: Optional[int] = None
    included_stems: Optional[List[str]] = None
    explicit_pairs: Optional[List[str]] = None
