"""Shared planning helpers for UI-shaped TRACE pipelines.

This module owns the small, serializable pipeline spec used by the web UI and
the CLI command generator.  Keeping command generation here prevents the React
app from learning shell syntax or duplicating CLI validation rules.
"""
from __future__ import annotations

import os
import shlex
from typing import Any, Dict, Literal, NamedTuple, Optional, Union

from pydantic import BaseModel, Field


TrainCacheMode = Literal["cached_video", "virtual"]
TrainCacheResolution = Literal[112, 144, 192, 224]
TrainResourceProfileId = Literal["auto", "low", "balanced", "high"]
ResourceProfileId = Literal["low", "balanced", "high"]


class PipelineSpecError(ValueError):
    """Raised when a pipeline spec cannot be run or rendered as CLI."""


class PipelineSteps(BaseModel):
    train: bool = False
    extra_test: bool = False
    infer: bool = False


class PipelineSelection(BaseModel):
    folder: str = ""
    pairs: list[str] = Field(default_factory=list)
    stems: list[str] = Field(default_factory=list)
    csv_by_stem: dict[str, str] = Field(default_factory=dict)


class PipelineResourceSettings(BaseModel):
    profile: ResourceProfileId = "balanced"
    batch_size: int = 4
    num_workers: int = 4
    decode_threads: int = 2
    prefetch_factor: int = 2


class PipelineResources(BaseModel):
    train: PipelineResourceSettings = Field(default_factory=PipelineResourceSettings)
    test: PipelineResourceSettings = Field(default_factory=PipelineResourceSettings)
    infer: PipelineResourceSettings = Field(default_factory=PipelineResourceSettings)


class PipelineSpec(BaseModel):
    steps: PipelineSteps = Field(default_factory=PipelineSteps)
    train_selection: PipelineSelection = Field(default_factory=PipelineSelection)
    test_selection: PipelineSelection = Field(default_factory=PipelineSelection)
    input_selection: PipelineSelection = Field(default_factory=PipelineSelection)
    model_size: Literal["small", "large"] = "small"
    model_dir: str = ""
    cache_mode: TrainCacheMode = "cached_video"
    cache_resolution: TrainCacheResolution = 144
    resource_profile: TrainResourceProfileId = "balanced"
    resources: PipelineResources = Field(default_factory=PipelineResources)
    epochs: int = 100
    val_start_epoch: int = 50
    val_interval: int = 10
    train_ratio: float = 0.8
    annotated_video: bool = False
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int = 42


class PipelineCommand(BaseModel):
    argv: list[str]
    command: str
    warnings: list[str] = Field(default_factory=list)


class TrainResourceProfile(NamedTuple):
    name: str
    id: str
    num_workers: int
    decode_threads: int
    prefetch_factor: int


TRAIN_RESOURCE_PROFILES: tuple[TrainResourceProfile, ...] = (
    TrainResourceProfile("Low", "low", 2, 1, 2),
    TrainResourceProfile("Balanced", "balanced", 4, 2, 2),
    TrainResourceProfile("High", "high", 8, 2, 2),
)


def resource_profile_by_id(profile_id: TrainResourceProfileId) -> TrainResourceProfile:
    if profile_id == "auto":
        return TRAIN_RESOURCE_PROFILES[1]
    for profile in TRAIN_RESOURCE_PROFILES:
        if profile.id == profile_id:
            return profile
    return TRAIN_RESOURCE_PROFILES[1]


def resource_settings_from_profile(profile_id: ResourceProfileId) -> PipelineResourceSettings:
    profile = resource_profile_by_id(profile_id)
    batch_size = {"low": 1, "balanced": 4, "high": 8}[profile_id]
    return PipelineResourceSettings(
        profile=profile_id,
        batch_size=batch_size,
        num_workers=profile.num_workers,
        decode_threads=profile.decode_threads,
        prefetch_factor=profile.prefetch_factor,
    )


def _clamped_int(value: int, fallback: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(min_value, min(max_value, parsed))


def _resource_loader_options(
    settings: PipelineResourceSettings,
    splits: list[str],
    *,
    include_batch: bool = False,
) -> Dict[str, Any]:
    default = resource_settings_from_profile(settings.profile)
    batch_size = _clamped_int(settings.batch_size, default.batch_size, 1, 256)
    num_workers = _clamped_int(settings.num_workers, default.num_workers, 1, 64)
    decode_threads = _clamped_int(settings.decode_threads, default.decode_threads, 1, 16)
    prefetch_factor = _clamped_int(settings.prefetch_factor, default.prefetch_factor, 1, 16)
    options: Dict[str, Any] = {}
    for split in splits:
        if include_batch:
            options[f"solver.{split}.batch_size"] = batch_size
        options[f"solver.{split}.num_workers"] = num_workers
        options[f"solver.{split}.prefetch_factor"] = prefetch_factor
        options[f"solver.{split}.persistent_workers"] = True
        options[f"dataset.{split}.pipeline.1.num_threads"] = decode_threads
    return options


def resource_profile_by_name(name: Optional[str]) -> TrainResourceProfile:
    normalised = (name or "").lower()
    for profile in TRAIN_RESOURCE_PROFILES:
        if profile.name.lower() == normalised:
            return profile
    return TRAIN_RESOURCE_PROFILES[1]


def resource_profile_cfg_options(
    profile: TrainResourceProfile,
    resolution: TrainCacheResolution = 144,
    model_size: Literal["small", "large"] = "small",
) -> Dict[str, Any]:
    size = [resolution, resolution]
    options: Dict[str, Any] = {
        "solver.train.num_workers": profile.num_workers,
        "solver.train.prefetch_factor": profile.prefetch_factor,
        "solver.train.persistent_workers": profile.num_workers > 0,
        "solver.val.num_workers": profile.num_workers,
        "solver.val.prefetch_factor": profile.prefetch_factor,
        "solver.val.persistent_workers": profile.num_workers > 0,
        "solver.test.num_workers": profile.num_workers,
        "solver.test.prefetch_factor": profile.prefetch_factor,
        "solver.test.persistent_workers": profile.num_workers > 0,
        "dataset.train.pipeline.1.num_threads": profile.decode_threads,
        "dataset.val.pipeline.1.num_threads": profile.decode_threads,
        "dataset.test.pipeline.1.num_threads": profile.decode_threads,
        "dataset.train.pipeline.1.resize": size,
        "dataset.val.pipeline.1.resize": size,
        "dataset.test.pipeline.1.resize": size,
        "dataset.val.pipeline.4.scale": size,
        "dataset.test.pipeline.4.scale": size,
    }
    if model_size == "large":
        options["dataset.train.pipeline.5.scale"] = size
    else:
        options["dataset.train.pipeline.4.scale"] = size
    return options


def train_resource_settings(spec: PipelineSpec) -> PipelineResourceSettings:
    if spec.resource_profile == "auto":
        return spec.resources.train
    if spec.resources.train.profile != spec.resource_profile:
        return resource_settings_from_profile(spec.resource_profile)
    return spec.resources.train


def train_resource_cfg_options(
    settings: PipelineResourceSettings,
    resolution: TrainCacheResolution = 144,
    model_size: Literal["small", "large"] = "small",
) -> Dict[str, Any]:
    size = [resolution, resolution]
    options: Dict[str, Any] = {
        "dataset.train.pipeline.1.resize": size,
        "dataset.val.pipeline.1.resize": size,
        "dataset.test.pipeline.1.resize": size,
        "dataset.val.pipeline.4.scale": size,
        "dataset.test.pipeline.4.scale": size,
    }
    options.update(_resource_loader_options(settings, ["train", "val", "test"]))
    if model_size == "large":
        options["dataset.train.pipeline.5.scale"] = size
    else:
        options["dataset.train.pipeline.4.scale"] = size
    return options


def eval_resource_cfg_options(settings: PipelineResourceSettings) -> Dict[str, Any]:
    return _resource_loader_options(settings, ["test"], include_batch=True)


def _first_pair_folder(selection: PipelineSelection) -> str:
    first_pair = selection.pairs[0] if selection.pairs else ""
    first_video = first_pair.split("=", 1)[0]
    if first_video and os.path.isabs(first_video):
        return os.path.dirname(first_video)
    return ""


def prep_selection(spec: PipelineSpec) -> PipelineSelection:
    return spec.train_selection if spec.steps.train else spec.test_selection


def prep_work_dir(spec: PipelineSpec) -> str:
    selection = prep_selection(spec)
    return selection.folder or _first_pair_folder(selection)


def prep_pairs(spec: PipelineSpec) -> list[str]:
    return list(prep_selection(spec).pairs)


def prep_cache_mode(spec: PipelineSpec) -> TrainCacheMode:
    if spec.steps.extra_test:
        return "cached_video"
    return spec.cache_mode


def validate_pipeline_spec(spec: PipelineSpec) -> None:
    steps = spec.steps
    has_any_step = steps.train or steps.extra_test or steps.infer
    needs_train_dataset = steps.train
    needs_test_dataset = steps.extra_test and not steps.train
    needs_model_load = not steps.train and (steps.extra_test or steps.infer)
    needs_input = steps.infer

    if not has_any_step:
        raise PipelineSpecError("Enable at least one pipeline step.")
    if needs_train_dataset and len(spec.train_selection.pairs) < 1:
        raise PipelineSpecError("Pick at least one training pair.")
    if needs_test_dataset and len(spec.test_selection.pairs) < 1:
        raise PipelineSpecError("Pick at least one test pair.")
    if needs_model_load and not spec.model_dir:
        raise PipelineSpecError("Model load folder is required when not training.")
    if needs_input and not spec.input_selection.folder:
        raise PipelineSpecError("Inference input folder is required.")
    if needs_input and len(spec.input_selection.stems) < 1:
        raise PipelineSpecError("Pick at least one inference video.")

    if steps.train:
        if not isinstance(spec.epochs, int) or spec.epochs < 1:
            raise PipelineSpecError("Total epochs must be at least 1.")
        if (
            not isinstance(spec.val_start_epoch, int)
            or spec.val_start_epoch < 0
            or spec.val_start_epoch >= spec.epochs
        ):
            raise PipelineSpecError("Validation start epoch must be between 0 and total epochs - 1.")
        if not isinstance(spec.val_interval, int) or spec.val_interval < 1:
            raise PipelineSpecError("Validation interval must be at least 1.")
        if not isinstance(spec.train_ratio, (int, float)) or spec.train_ratio <= 0 or spec.train_ratio >= 1:
            raise PipelineSpecError("Train/val ratio must be between 0 and 1 (exclusive).")

    if (steps.train or steps.extra_test) and not prep_work_dir(spec):
        raise PipelineSpecError("Dataset work folder is required.")

    resource_checks = []
    if steps.train and spec.resource_profile != "auto":
        resource_checks.append(("Train", train_resource_settings(spec), False))
    if steps.extra_test:
        resource_checks.append(("Test", spec.resources.test, True))
    if steps.infer:
        resource_checks.append(("Inference", spec.resources.infer, True))
    for stage, settings, include_batch in resource_checks:
        if include_batch and settings.batch_size < 1:
            raise PipelineSpecError(f"{stage} batch size must be at least 1.")
        if settings.num_workers < 1:
            raise PipelineSpecError(f"{stage} workers must be at least 1.")
        if settings.decode_threads < 1:
            raise PipelineSpecError(f"{stage} decode threads must be at least 1.")
        if settings.prefetch_factor < 1:
            raise PipelineSpecError(f"{stage} prefetch must be at least 1.")


def _fmt_number(value: Union[int, float]) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _append_resource_args(
    argv: list[str],
    *,
    settings: PipelineResourceSettings,
    profile_flag: str,
    prefix: str,
    include_batch: bool,
) -> None:
    argv.extend([profile_flag, settings.profile])
    default = resource_settings_from_profile(settings.profile)
    if include_batch and settings.batch_size != default.batch_size:
        argv.extend([f"--{prefix}-batch-size", str(settings.batch_size)])
    if settings.num_workers != default.num_workers:
        argv.extend([f"--{prefix}-workers", str(settings.num_workers)])
    if settings.decode_threads != default.decode_threads:
        argv.extend([f"--{prefix}-decode-threads", str(settings.decode_threads)])
    if settings.prefetch_factor != default.prefetch_factor:
        argv.extend([f"--{prefix}-prefetch", str(settings.prefetch_factor)])


def build_pipeline_argv(spec: PipelineSpec) -> list[str]:
    validate_pipeline_spec(spec)
    argv = ["trace", "pipeline", "--model", spec.model_size]

    if spec.steps.train:
        argv.append("--train")
    if spec.steps.extra_test:
        argv.append("--extra-test")
    if spec.steps.infer:
        argv.append("--infer")

    if not spec.steps.train and (spec.steps.extra_test or spec.steps.infer):
        argv.extend(["--model-dir", spec.model_dir])

    if spec.steps.train or spec.steps.extra_test:
        argv.extend(["--work-dir", prep_work_dir(spec)])
        pairs = prep_pairs(spec)
        if pairs:
            argv.extend(["--pairs", *pairs])
        argv.extend([
            "--cache-mode", prep_cache_mode(spec),
            "--cache-resolution", str(spec.cache_resolution),
            "--train-ratio", _fmt_number(spec.train_ratio),
        ])

    if spec.steps.train:
        train_settings = train_resource_settings(spec)
        argv.extend([
            "--epochs", str(spec.epochs),
            "--val-start-epoch", str(spec.val_start_epoch),
            "--val-interval", str(spec.val_interval),
        ])
        if spec.resource_profile == "auto":
            argv.extend(["--resource-profile", "auto"])
        else:
            _append_resource_args(
                argv,
                settings=train_settings,
                profile_flag="--resource-profile",
                prefix="train",
                include_batch=False,
            )

    if spec.steps.extra_test:
        _append_resource_args(
            argv,
            settings=spec.resources.test,
            profile_flag="--test-resource-profile",
            prefix="test",
            include_batch=True,
        )

    if spec.steps.infer:
        _append_resource_args(
            argv,
            settings=spec.resources.infer,
            profile_flag="--infer-resource-profile",
            prefix="infer",
            include_batch=True,
        )
        argv.extend(["--input", spec.input_selection.folder])
        if spec.input_selection.stems:
            argv.extend(["--include-stems", *spec.input_selection.stems])
        if spec.annotated_video:
            argv.append("--annotated-video")
        if spec.threshold != 0.0:
            argv.extend(["--threshold", _fmt_number(spec.threshold)])

    if spec.seed != 42:
        argv.extend(["--seed", str(spec.seed)])

    return argv


def build_pipeline_command(spec: PipelineSpec) -> PipelineCommand:
    argv = build_pipeline_argv(spec)
    return PipelineCommand(argv=argv, command=shlex.join(argv), warnings=[])


def _resource_settings_from_cli_args(
    args: Any,
    *,
    prefix: str,
    profile_attr: str,
    include_batch: bool,
) -> PipelineResourceSettings:
    profile = getattr(args, profile_attr, "balanced") or "balanced"
    if profile == "auto":
        profile = "balanced"
    settings = resource_settings_from_profile(profile)
    if include_batch:
        batch_size = getattr(args, f"{prefix}_batch_size", None)
        if batch_size is not None:
            settings.batch_size = batch_size
    workers = getattr(args, f"{prefix}_workers", None)
    if workers is not None:
        settings.num_workers = workers
    decode_threads = getattr(args, f"{prefix}_decode_threads", None)
    if decode_threads is not None:
        settings.decode_threads = decode_threads
    prefetch = getattr(args, f"{prefix}_prefetch", None)
    if prefetch is not None:
        settings.prefetch_factor = prefetch
    return settings


def spec_from_cli_args(args: Any) -> PipelineSpec:
    steps = PipelineSteps(
        train=bool(getattr(args, "train", False)),
        extra_test=bool(getattr(args, "extra_test", False)),
        infer=bool(getattr(args, "infer", False)),
    )
    pairs = list(getattr(args, "explicit_pairs", None) or [])
    work_dir = getattr(args, "work_dir", None) or ""
    train_selection = PipelineSelection(folder=work_dir, pairs=pairs if steps.train else [])
    test_selection = PipelineSelection(folder=work_dir, pairs=[] if steps.train else pairs)
    input_selection = PipelineSelection(
        folder=getattr(args, "input", None) or "",
        stems=list(getattr(args, "include_stems", None) or []),
    )

    train_resource_profile = getattr(args, "resource_profile", "balanced")
    resources = PipelineResources(
        train=_resource_settings_from_cli_args(
            args,
            prefix="train",
            profile_attr="resource_profile",
            include_batch=False,
        ),
        test=_resource_settings_from_cli_args(
            args,
            prefix="test",
            profile_attr="test_resource_profile",
            include_batch=True,
        ),
        infer=_resource_settings_from_cli_args(
            args,
            prefix="infer",
            profile_attr="infer_resource_profile",
            include_batch=True,
        ),
    )

    return PipelineSpec(
        steps=steps,
        train_selection=train_selection,
        test_selection=test_selection,
        input_selection=input_selection,
        model_size=getattr(args, "model", "small"),
        model_dir=getattr(args, "model_dir", None) or "",
        cache_mode=getattr(args, "cache_mode", "cached_video"),
        cache_resolution=getattr(args, "cache_resolution", 144),
        resource_profile=train_resource_profile,
        resources=resources,
        epochs=getattr(args, "epochs", 100),
        val_start_epoch=getattr(args, "val_start_epoch", 50),
        val_interval=getattr(args, "val_interval", 10),
        train_ratio=getattr(args, "train_ratio", 0.8),
        annotated_video=bool(getattr(args, "annotated_video", False)),
        threshold=getattr(args, "threshold", 0.0),
        seed=getattr(args, "seed", 42),
    )
