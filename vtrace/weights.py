"""Auto-download for pretrained backbone weights.

Configs in `configs/*.py` reference backbone weights as a relative path
like `pretrained/vit-large-p16_videomaev2-k400.pth`. When the file is missing,
we resolve the basename against the registry below and pull it from the
project's GitHub Release into a user-level cache (`~/.vtrace/pretrained/`).

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

# Additional hosts, tried in order after the registered URL fails. Each entry is a
# template with a `{name}` placeholder, so a Zenodo deposit of the same filenames
# needs one line here and no change to the registry:
#   "https://zenodo.org/records/<RECORD_ID>/files/{name}?download=1"
_MIRROR_URLS: tuple[str, ...] = ()

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
    # ── Released CalMS21 checkpoints + their two pretrained bases. ──
    # Any config or --checkpoint referencing these basenames auto-downloads on miss.
    "vitB_videomaev2_k400.pth": (
        f"{_RELEASE_BASE}/vitB_videomaev2_k400.pth",
        "3e7f93b2be64c7c1cbe4587a1561d7c7a277eab3bad7f87dad7a4b52ec76f367",
        173101070,
    ),
    "vjepa2_vitl_encoder.pth": (
        f"{_RELEASE_BASE}/vjepa2_vitl_encoder.pth",
        "3fa5eed32ea68b1904a6144e14ec6e42947074b63b2405cfd094c71f5d1440f0",
        1215541535,
    ),
    "calms21_vjepa2_teacher95.pth": (
        f"{_RELEASE_BASE}/calms21_vjepa2_teacher95.pth",
        "ca57270a35c164cd2ad614bdffe0553f1b04f3ec5410abac8589bf90f02accb3",
        1215665755,
    ),
    "calms21_vitB_distilled_best.pth": (
        f"{_RELEASE_BASE}/calms21_vitB_distilled_best.pth",
        "2964f87d303e7b27264ba02e5d270e610037fc16a7087f2509c55997e5a0dcd1",
        824375013,
    ),
}

MODEL_WEIGHT_FILES: dict[str, str] = {
    "small": "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
    "large": "vit-large-p16_videomaev2-k400.pth",
}


def _user_dir(name: str) -> Path:
    """`~/.vtrace/<name>`, or the pre-rename `~/.trace/<name>` when that is the one
    already holding data — renaming the tool should not orphan a downloaded cache."""
    home = Path.home()
    new = home / ".vtrace" / name
    old = home / ".trace" / name
    return old if (old.exists() and not new.exists()) else new


def cache_dir() -> Path:
    root = os.environ.get("TRACE_WEIGHTS_DIR")
    return Path(root) if root else _user_dir("pretrained")


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
    return str(fetch(name, url, sha256, size))


def _urls_for(name: str, registered: str) -> list[str]:
    """The registered URL first, then every configured mirror."""
    urls = [registered]
    for template in _MIRROR_URLS:
        mirror = template.format(name=name)
        if mirror not in urls:
            urls.append(mirror)
    return urls


def fetch(name: str, url: str, sha256: str, size: Optional[int] = None,
          dest_dir: Optional[Path] = None) -> Path:
    """Return a verified local copy of `name`, downloading it if the cache misses.

    Tries the registered URL then each mirror, so a Zenodo deposit can stand in for
    the GitHub release (or vice versa) without touching call sites.
    """
    cached = (dest_dir or cache_dir()) / name
    if cached.is_file() and _sha256(cached) == sha256:
        return cached

    cached.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for candidate in _urls_for(name, url):
        print(f"[weights] {name}: downloading from {candidate}", file=sys.stderr)
        try:
            _download(candidate, cached, expected_size=size)
        except RuntimeError as e:
            errors.append(str(e))
            continue

        actual = _sha256(cached)
        if actual != sha256:
            cached.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch for {name}: expected {sha256}, got {actual}. "
                f"Re-run to retry, or download manually from {candidate}."
            )
        print(f"[weights] saved to {cached}", file=sys.stderr)
        return cached

    raise RuntimeError(
        f"Could not download {name} from any host:\n  " + "\n  ".join(errors)
    )


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
