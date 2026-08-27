"""Dataloader resource profiles, and the `--cfg-options` each step needs.

A profile is a name for one set of dataloader knobs — batch size, worker count,
decode threads, prefetch depth — so a user picks `low` / `balanced` / `high`
instead of four numbers whose safe combinations they would have to know.

Split out of the retired `pipeline_plan` module, which wrapped these in a spec
object describing a whole `vtrace pipeline` run. The steps are chained at the
prompt now, so what survives is the part that was never about the pipeline: the
profiles themselves, and turning one into config overrides.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, NamedTuple, Optional

from vtrace.proxy_geometry import decode_thread_cfg_options, input_resize_cfg_options


# Model input resolution. Every value must be a multiple of 16: PatchEmbed is
# an unpadded stride-16 Conv3d, so a non-multiple silently truncates the
# right/bottom edge of every frame. 256 is V-JEPA2-L's native grid.
InputResolution = Literal[112, 144, 160, 192, 224, 256]
INPUT_RESOLUTIONS = (112, 144, 160, 192, 224, 256)

ResourceProfileId = Literal["low", "balanced", "high"]
RESOURCE_PROFILE_IDS = ("low", "balanced", "high")
DEFAULT_RESOURCE_PROFILE = "balanced"

# Splits whose loaders a training run drives, and the one an evaluation or
# prediction run drives.
TRAIN_SPLITS = ["train", "val", "test"]
EVAL_SPLITS = ["test"]


class ResourceProfile(NamedTuple):
    name: str
    id: str
    batch_size: int
    num_workers: int
    decode_threads: int
    prefetch_factor: int


RESOURCE_PROFILES: tuple[ResourceProfile, ...] = (
    ResourceProfile("Low", "low", 1, 2, 1, 2),
    ResourceProfile("Balanced", "balanced", 4, 4, 2, 2),
    ResourceProfile("High", "high", 8, 8, 2, 2),
)


def profile_by_id(profile_id: Optional[str]) -> ResourceProfile:
    """The named profile, falling back to balanced for anything unrecognised."""
    for profile in RESOURCE_PROFILES:
        if profile.id == profile_id:
            return profile
    return RESOURCE_PROFILES[1]


def profile_by_name(name: Optional[str]) -> ResourceProfile:
    """Look a profile up by display name — what the train tuner reports."""
    normalised = (name or "").lower()
    for profile in RESOURCE_PROFILES:
        if profile.name.lower() == normalised:
            return profile
    return RESOURCE_PROFILES[1]


def _loader_options(
    profile: ResourceProfile,
    splits: list[str],
    *,
    include_batch: bool,
    cfg: Any,
) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    for split in splits:
        if include_batch:
            options[f"solver.{split}.batch_size"] = profile.batch_size
        options[f"solver.{split}.num_workers"] = profile.num_workers
        options[f"solver.{split}.prefetch_factor"] = profile.prefetch_factor
        options[f"solver.{split}.persistent_workers"] = True
    options.update(decode_thread_cfg_options(cfg, profile.decode_threads, splits))
    return options


def train_cfg_options(
    profile: ResourceProfile,
    cfg: Any,
    resolution: Optional[int] = None,
) -> Dict[str, Any]:
    """Overrides for a training run: every loader, plus an optional resize.

    Batch size is left alone. A training batch is bounded by what fits in VRAM
    alongside the gradients, which the profile cannot know; the tuner exists for
    that question.
    """
    options: Dict[str, Any] = (
        dict(input_resize_cfg_options(cfg, resolution)) if resolution else {}
    )
    options.update(_loader_options(profile, TRAIN_SPLITS, include_batch=False, cfg=cfg))
    return options


def eval_cfg_options(profile: ResourceProfile, cfg: Any) -> Dict[str, Any]:
    """Overrides for an evaluation or prediction run: the test loader only."""
    return _loader_options(profile, EVAL_SPLITS, include_batch=True, cfg=cfg)
