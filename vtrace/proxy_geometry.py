"""Derive the decode-proxy geometry a config's data pipeline needs.

TRACE decodes training windows from a downscaled, frame-aligned *proxy* of each
source video rather than from the full-resolution original (see
``lossless-video-pipeline.md (archived)``). The proxy's pixel geometry has to match
what the pipeline is going to do with the frames, so it is derived from the
pipeline itself instead of being configured separately.

Two pipeline styles exist and they want different things:

* **square squash** — ``VideoInit(resize=(R, R))`` asks decord to decode
  straight to ``R x R``, destroying the aspect ratio, and ``VideoBatchResize``
  then no-ops. A square proxy reproduces exactly those pixels.
* **aspect preserving** — ``VideoResize(scale=(-1, S))`` followed by
  ``VideoCenterCrop`` / ``VideoRandomResizedCrop``. A squashed proxy would
  change the crop's field of view and make the crop's aspect-ratio sampling
  meaningless, so the proxy has to keep the source aspect ratio.

Kept dependency-free (stdlib only) so ``tools/``, the CLI, ``data_prep`` and the
server can all import it without pulling in cv2/numpy/torch.
"""

from typing import Any, Dict, Iterable, NamedTuple, Optional, Sequence, Tuple

# PatchEmbed is an unpadded stride-16 Conv3d, so a spatial size that is not a
# multiple of the patch size silently truncates the right/bottom edge.
PATCH_ALIGN = 16

DEFAULT_SPLITS = ("train", "val", "test")

# Pipeline steps that resize/crop, and the field carrying their target size.
_SQUARE_STEP = "VideoInit"
_SIZE_STEPS = {
    "VideoInit": "resize",
    "VideoResize": "scale",
    "VideoBatchResize": "scale",
    "VideoCenterCrop": "crop_size",
}


class ProxyGeometry(NamedTuple):
    """Pixel geometry of a decode proxy.

    Attributes:
        short_side: Target short side in pixels (also the square side).
        square: True to squash to ``short_side x short_side``, False to scale
            the short side and let the long side follow the source aspect.
    """

    short_side: int
    square: bool

    @property
    def vf(self) -> str:
        """The ffmpeg ``-vf`` scale expression for this geometry."""
        if self.square:
            return f"scale={self.short_side}:{self.short_side}"
        # -2 keeps the long side even, which yuv420p requires.
        return f"scale=-2:{self.short_side}"

    @property
    def key(self) -> str:
        """Short cache-key fragment, e.g. ``s144_sq``."""
        return f"s{self.short_side}_{'sq' if self.square else 'ar'}"


def align_up(value, multiple: int = PATCH_ALIGN) -> int:
    """Round ``value`` up to the next multiple of ``multiple`` (min one step)."""
    value = int(value)
    if value <= 0:
        return multiple
    return -(-value // multiple) * multiple


def _step_get(step, field, default=None):
    """Read a field from a pipeline step (dict or attribute-style Config)."""
    if isinstance(step, dict):
        return step.get(field, default)
    return getattr(step, field, default)


def _short_side_demand(step_type: str, value) -> Optional[int]:
    """Short side a single resize/crop step needs from the decoded frame."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (list, tuple)) and value:
        dims = [int(v) for v in value]
        positive = [v for v in dims if v > 0]
        if not positive:
            return None
        if step_type == "VideoResize" and any(v <= 0 for v in dims):
            # scale=(-1, S) / (S, -1): the positive entry is the short side.
            return positive[0]
        if len(positive) == 2:
            # An exact (H, W) target. What it needs from the decoded frame is a
            # short side of min(H, W), not max: the long side arrives for free
            # from the aspect ratio. Square targets have min == max, which is why
            # `max` was indistinguishable until a non-square config existed --
            # for (252, 448) it demanded a 448 short side and so a proxy roughly
            # three times the pixels the model ever sees.
            return min(positive)
        return max(positive)
    return None


def pipeline_geometry(pipeline: Iterable, *, fallback: int = 144) -> Optional[ProxyGeometry]:
    """Geometry one split's pipeline needs, or None if it has no resize step.

    ``square`` is True when the pipeline decodes straight to a fixed size via
    ``VideoInit(resize=...)``, which already discards the aspect ratio.
    """
    if pipeline is None:
        return None

    demand = 0
    square = False
    saw_step = False

    for step in pipeline:
        step_type = _step_get(step, "type")
        field = _SIZE_STEPS.get(step_type)
        if field is None:
            continue
        value = _step_get(step, field)
        if step_type == _SQUARE_STEP and value is not None:
            square = True
        short_side = _short_side_demand(step_type, value)
        if short_side is None:
            continue
        saw_step = True
        demand = max(demand, short_side)

    if not saw_step:
        return None
    return ProxyGeometry(align_up(demand or fallback), square)


def config_geometry(
    cfg,
    splits: Sequence[str] = DEFAULT_SPLITS,
    *,
    fallback: int = 144,
) -> ProxyGeometry:
    """Geometry that satisfies every split's pipeline.

    Takes the largest short side any split asks for. Stays square only when
    every split that declares a geometry is square-style — an aspect-preserving
    proxy is a strict superset, so a mixed config falls back to aspect.
    """
    geometries = []
    dataset = getattr(cfg, "dataset", None)
    for split in splits:
        split_cfg = getattr(dataset, split, None) if dataset is not None else None
        if split_cfg is None:
            continue
        geometry = pipeline_geometry(_step_get(split_cfg, "pipeline"), fallback=fallback)
        if geometry is not None:
            geometries.append(geometry)

    if not geometries:
        return ProxyGeometry(align_up(fallback), True)
    return ProxyGeometry(
        max(g.short_side for g in geometries),
        all(g.square for g in geometries),
    )


def _resize_value(step_type: str, current, resolution: int):
    """New value for a resize/crop step at ``resolution``, matching its shape."""
    if step_type == "VideoResize" and isinstance(current, (list, tuple)) and len(current) == 2:
        # Preserve a (-1, S) / (S, -1) short-side spec rather than squashing it.
        dims = [int(v) for v in current]
        if dims[0] <= 0:
            return [-1, resolution]
        if dims[1] <= 0:
            return [resolution, -1]
    if isinstance(current, (int, float)):
        return resolution
    return [resolution, resolution]


def _pipeline_cfg_options(
    cfg,
    splits: Sequence[str],
    step_types: Dict[str, str],
    make_value,
) -> Dict[str, Any]:
    """Emit ``dataset.{split}.pipeline.{i}.{field}`` overrides by step type.

    Locating steps by type rather than by position keeps these overrides valid
    when a config inserts an extra step (the shipped presets add
    ``VideoTemporalAugment``, which is why the index-based version needed a
    per-model special case).
    """
    options: Dict[str, Any] = {}
    dataset = getattr(cfg, "dataset", None)
    if dataset is None:
        return options

    for split in splits:
        split_cfg = getattr(dataset, split, None)
        if split_cfg is None:
            continue
        pipeline = _step_get(split_cfg, "pipeline")
        if not pipeline:
            continue
        for index, step in enumerate(pipeline):
            step_type = _step_get(step, "type")
            field = step_types.get(step_type)
            if field is None:
                continue
            current = _step_get(step, field)
            if current is None:
                continue
            options[f"dataset.{split}.pipeline.{index}.{field}"] = make_value(
                step_type, current
            )
    return options


def input_resize_cfg_options(
    cfg,
    resolution: int,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> Dict[str, Any]:
    """Config overrides retargeting every resize/crop step to ``resolution``."""
    resolution = align_up(resolution)
    return _pipeline_cfg_options(
        cfg,
        splits,
        _SIZE_STEPS,
        lambda step_type, current: _resize_value(step_type, current, resolution),
    )


def decode_thread_cfg_options(
    cfg,
    num_threads: int,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> Dict[str, Any]:
    """Config overrides setting ``VideoInit.num_threads`` on every split."""
    return _pipeline_cfg_options(
        cfg,
        splits,
        {"VideoInit": "num_threads"},
        lambda step_type, current: int(num_threads),
    )


