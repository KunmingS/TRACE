"""Auto-download for pretrained backbone weights.

Configs in `configs/*.py` reference backbone weights as a relative path
like `pretrained/vit-large-p16_videomaev2-k400.pth`. When the file is missing,
we resolve the basename against the registry below and pull it from the
project's GitHub Release into a user-level cache (`~/.trace/pretrained/`).

The release tag is fixed (`weights`); add new entries here when publishing
new backbone files.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


_RELEASE_BASE = "https://github.com/KunmingS/TRACE/releases/download/weights"

# basename -> (url, sha256, size_bytes)
_REGISTRY: dict[str, tuple[str, str, int]] = {
    "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth": (
        f"{_RELEASE_BASE}/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
        "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251",
        90605819,
    ),
    "vit-large-p16_videomaev2-k400.pth": (
        f"{_RELEASE_BASE}/vit-large-p16_videomaev2-k400.pth",
        "49b2dadc3fa55cc2c682793858dc0adb9224016da8aed6844f128f5f3d19c4f1",
        607765274,
    ),
}

MODEL_WEIGHT_FILES: dict[str, str] = {
    "small": "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
    "large": "vit-large-p16_videomaev2-k400.pth",
}


def cache_dir() -> Path:
    root = os.environ.get("TRACE_WEIGHTS_DIR")
    return Path(root) if root else Path.home() / ".trace" / "pretrained"


def model_weight_choices(include_all: bool = True) -> tuple[str, ...]:
    choices = tuple(MODEL_WEIGHT_FILES)
    return ("all", *choices) if include_all else choices


def model_weight_names(selection: str = "all") -> list[str]:
    if selection == "all":
        return list(MODEL_WEIGHT_FILES.values())
    if selection not in MODEL_WEIGHT_FILES:
        valid = ", ".join(model_weight_choices())
        raise ValueError(f"Unknown model weight selection '{selection}'. Choose one of: {valid}")
    return [MODEL_WEIGHT_FILES[selection]]


def download_model_weights(selection: str = "all") -> list[str]:
    """Download and verify the requested model weights, returning local paths."""
    return [resolve(name) for name in model_weight_names(selection)]


def resolve(path: str) -> str:
    """Return a usable filesystem path for `path`, downloading if needed.

    - If `path` already exists, return it unchanged.
    - Else, if its basename is in the registry, ensure the cached copy
      exists (downloading + verifying SHA256 on miss) and return that path.
    - Else, return `path` unchanged so the caller raises its own FileNotFoundError.
    """
    if os.path.isfile(path):
        return path

    name = os.path.basename(path)
    spec = _REGISTRY.get(name)
    if spec is None:
        return path

    url, sha256, size = spec
    cached = cache_dir() / name
    if cached.is_file() and _sha256(cached) == sha256:
        return str(cached)

    cached.parent.mkdir(parents=True, exist_ok=True)
    print(f"[weights] {name} not found locally; downloading from {url}", file=sys.stderr)
    _download(url, cached, expected_size=size)

    actual = _sha256(cached)
    if actual != sha256:
        cached.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA256 mismatch for {name}: expected {sha256}, got {actual}. "
            f"Re-run to retry, or download manually from {url}."
        )
    print(f"[weights] saved to {cached}", file=sys.stderr)
    return str(cached)


def _download(url: str, dest: Path, expected_size: Optional[int] = None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as resp:
            total = expected_size or int(resp.headers.get("Content-Length") or 0)
            with open(tmp, "wb") as f:
                _copy_with_progress(resp, f, total, label=dest.name)
        tmp.replace(dest)
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _copy_with_progress(src, dst, total: int, label: str, chunk: int = 1024 * 1024) -> None:
    try:
        from tqdm import tqdm
        bar = tqdm(total=total or None, unit="B", unit_scale=True, desc=label, file=sys.stderr)
        try:
            while True:
                buf = src.read(chunk)
                if not buf:
                    break
                dst.write(buf)
                bar.update(len(buf))
        finally:
            bar.close()
    except ImportError:
        shutil.copyfileobj(src, dst, length=chunk)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
