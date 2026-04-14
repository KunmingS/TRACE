"""Pydantic models for TRACE job management."""
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class JobType(str, Enum):
    TRAIN = "train"
    TEST = "test"
    INFER = "infer"
    PREP = "prep"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobInfo(BaseModel):
    job_id: str
    job_type: JobType
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
    args: dict = {}


class TrainRequest(BaseModel):
    config_path: str
    nproc: int = 1
    seed: int = 42
    exp_id: int = 0
    resume: Optional[str] = None
    not_eval: bool = False
    disable_deterministic: bool = False
    dataset_dir: Optional[str] = None
    annotation_path: Optional[str] = None
    class_map: Optional[str] = None
    pretrained: Optional[str] = None
    cfg_options: Optional[dict] = None


class TestRequest(BaseModel):
    config_path: str
    checkpoint: str
    nproc: int = 1
    seed: int = 42
    exp_id: int = 0
    not_eval: bool = False
    profile: bool = False
    auto_tune: bool = True
    dataset_dir: Optional[str] = None
    annotation_path: Optional[str] = None
    class_map: Optional[str] = None
    cfg_options: Optional[dict] = None


class InferRequest(BaseModel):
    config_path: str
    checkpoint: str
    input: str
    class_map: str
    output: Optional[str] = None
    seed: int = 42
    exp_id: int = 0
    profile: bool = False
    auto_tune: bool = True
    cfg_options: Optional[dict] = None


class PrepRequest(BaseModel):
    dataset_path: str
    clip_frames: int = 768
    train_ratio: float = 0.8
