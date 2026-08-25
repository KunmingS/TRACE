"""The spec `vtrace pipeline` runs, and the config overrides each step needs.

`spec_from_cli_args` turns parsed CLI flags into a `PipelineSpec`,
`validate_pipeline_spec` rejects the combinations that cannot run, and the
`*_resource_*` helpers turn the chosen resource profile into `--cfg-options`
for the step being launched.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Literal, NamedTuple, Optional

from pydantic import BaseModel, Field

from vtrace.proxy_geometry import decode_thread_cfg_options, input_resize_cfg_options


# Model input resolution. Every value must be a multiple of 16: PatchEmbed is
# an unpadded stride-16 Conv3d, so a non-multiple silently truncates the
# right/bottom edge of every frame. 256 is V-JEPA2-L's native grid.
InputResolution = Literal[112, 144, 160, 192, 224, 256]
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


class PipelineSpec(BaseModel):
    steps: PipelineSteps = Field(default_factory=PipelineSteps)
    train_selection: PipelineSelection = Field(default_factory=PipelineSelection)
    test_selection: PipelineSelection = Field(default_factory=PipelineSelection)
    input_selection: PipelineSelection = Field(default_factory=PipelineSelection)
    # Name of a `MODEL_CONFIGS` preset; the CLI validates it against that dict.
    model: str = ""
    # Explicit config path; overrides `model` when set.
    config: str = ""
    model_dir: str = ""
    # Overrides both the model input resize and the decode-proxy geometry.
    # None means "whatever the model config already asks for".
    input_resolution: Optional[InputResolution] = None
    resource_profile: TrainResourceProfileId = "balanced"
    epochs: int = 100
    val_start_epoch: int = 50
    val_interval: int = 10
    train_ratio: float = 0.8
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int = 42


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


def _resource_loader_options(
    settings: PipelineResourceSettings,
    splits: list[str],
    *,
    include_batch: bool = False,
    cfg: Any,
) -> Dict[str, Any]:
    batch_size = settings.batch_size
    num_workers = settings.num_workers
    prefetch_factor = settings.prefetch_factor
    options: Dict[str, Any] = {}
    for split in splits:
        if include_batch:
            options[f"solver.{split}.batch_size"] = batch_size
        options[f"solver.{split}.num_workers"] = num_workers
        options[f"solver.{split}.prefetch_factor"] = prefetch_factor
        options[f"solver.{split}.persistent_workers"] = True
    options.update(decode_thread_cfg_options(cfg, settings.decode_threads, splits))
    return options


def resource_profile_by_name(name: Optional[str]) -> TrainResourceProfile:
    normalised = (name or "").lower()
    for profile in TRAIN_RESOURCE_PROFILES:
        if profile.name.lower() == normalised:
            return profile
    return TRAIN_RESOURCE_PROFILES[1]


def train_resource_settings(spec: PipelineSpec) -> PipelineResourceSettings:
    return resource_settings_from_profile(_concrete_profile(spec))


def eval_resource_settings(spec: PipelineSpec) -> PipelineResourceSettings:
    """Settings for the test/predict steps.

    They follow the same profile as training: `auto` only benchmarks the train
    dataloader, so evaluation falls back to the balanced defaults.
    """
    return resource_settings_from_profile(_concrete_profile(spec))


def _concrete_profile(spec: PipelineSpec) -> ResourceProfileId:
    return "balanced" if spec.resource_profile == "auto" else spec.resource_profile


def train_resource_cfg_options(
    settings: PipelineResourceSettings,
    resolution: Optional[InputResolution],
    cfg: Any,
) -> Dict[str, Any]:
    options: Dict[str, Any] = (
        dict(input_resize_cfg_options(cfg, resolution)) if resolution else {}
    )
    options.update(_resource_loader_options(settings, ["train", "val", "test"], cfg=cfg))
    return options


def eval_resource_cfg_options(
    settings: PipelineResourceSettings, cfg: Any
) -> Dict[str, Any]:
    return _resource_loader_options(settings, ["test"], include_batch=True, cfg=cfg)


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

    return PipelineSpec(
        steps=steps,
        train_selection=train_selection,
        test_selection=test_selection,
        input_selection=input_selection,
        model=getattr(args, "model", "") or "",
        config=getattr(args, "config", None) or "",
        model_dir=getattr(args, "model_dir", None) or "",
        input_resolution=getattr(args, "input_resolution", None),
        resource_profile=getattr(args, "resource_profile", "balanced"),
        epochs=getattr(args, "epochs", 100),
        val_start_epoch=getattr(args, "val_start_epoch", 50),
        val_interval=getattr(args, "val_interval", 10),
        train_ratio=getattr(args, "train_ratio", 0.8),
        threshold=getattr(args, "threshold", 0.0),
        seed=getattr(args, "seed", 42),
    )
