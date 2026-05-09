"""Helpers for TRACE model artifact directories."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_timestamped_dir(parent, prefix, *, now=None):
    """Create ``<parent>/<prefix>_YYYYMMDD_HHMMSS`` and return its path."""
    parent_path = Path(parent).expanduser().resolve()
    parent_path.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base_name = f"{prefix}_{timestamp}"

    for index in range(1000):
        name = base_name if index == 0 else f"{base_name}_{index + 1}"
        candidate = parent_path / name
        try:
            candidate.mkdir()
            return str(candidate)
        except FileExistsError:
            continue

    raise FileExistsError(f"Could not create a unique {prefix} directory under {parent_path}")


def create_model_dir(work_dir, *, now=None):
    return create_timestamped_dir(work_dir, "model", now=now)


def create_eval_dir(model_dir, *, now=None):
    return create_timestamped_dir(model_dir, "eval", now=now)


def create_predict_dir(parent, *, now=None):
    """Create a prediction run directory under ``parent``."""
    return create_timestamped_dir(parent, "predict", now=now)


def predict_parent_for_input(input_path):
    """Return the directory that should own prediction artifacts for input."""
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        return path
    return path.parent


def create_predict_dir_for_input(input_path, *, now=None):
    return create_timestamped_dir(predict_parent_for_input(input_path), "predict", now=now)


def resolve_model_dir(model_dir):
    """Return canonical paths inside a trained model directory."""
    model_path = Path(model_dir).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    best_pth = model_path / "best.pth"
    classmap = model_path / "classmap.txt"
    config_file = model_path / "config.txt"
    dataset_json = model_path / "dataset.json"

    missing = [
        str(path)
        for path in (best_pth, classmap, config_file)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Model directory is missing required file(s): " + ", ".join(missing)
        )

    config_path = config_file.read_text(encoding="utf-8").strip()
    if not config_path:
        raise ValueError(f"Empty config.txt in model directory: {model_path}")

    return {
        "model_dir": str(model_path),
        "checkpoint": str(best_pth),
        "class_map": str(classmap),
        "config_path": config_path,
        "dataset_json": str(dataset_json) if dataset_json.is_file() else None,
    }
