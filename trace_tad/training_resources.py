"""Training resource estimates for selected TRACE video/CSV pairs.

The estimates here are intentionally lightweight: they read container metadata
and annotation CSVs, then use conservative heuristics. The frontend uses this
module to show resource tradeoffs before any long-running job starts, so users
can pick an explicit cache mode, resolution, model size, and dataloader profile.
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional


BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024 * BYTES_PER_MB
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

TRAIN_RESOURCE_PROFILES = [
    dict(id="low", name="Low", num_workers=2, decode_threads=1, prefetch_factor=2),
    dict(id="balanced", name="Balanced", num_workers=4, decode_threads=2, prefetch_factor=2),
    dict(id="high", name="High", num_workers=8, decode_threads=2, prefetch_factor=2),
]
TRAIN_RESOLUTIONS = (112, 144, 192, 224)


def _csv_dict_reader(file_obj):
    return csv.DictReader(line for line in file_obj if not line.lstrip().startswith("#"))


def _resolve_dataset_file(dataset_path: str, file_path: str) -> str:
    return file_path if os.path.isabs(file_path) else os.path.join(dataset_path, file_path)


def _normalise_explicit_pairs(dataset_path: str, explicit_pairs: Optional[list[str]]):
    if not explicit_pairs:
        return None

    pairs = []
    seen_videos = set()
    for spec in explicit_pairs:
        spec = str(spec).strip()
        if not spec:
            continue
        if "=" not in spec:
            raise ValueError(f"Invalid pair spec '{spec}'. Use VIDEO_PATH=CSV_PATH.")
        video_spec, csv_spec = (part.strip() for part in spec.split("=", 1))
        if not video_spec or not csv_spec:
            raise ValueError(f"Invalid pair spec '{spec}'. Use VIDEO_PATH=CSV_PATH.")
        video_path = os.path.abspath(_resolve_dataset_file(dataset_path, video_spec))
        csv_path = os.path.abspath(_resolve_dataset_file(dataset_path, csv_spec))
        if os.path.splitext(video_path)[1].lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Pair video must use one of {sorted(VIDEO_EXTENSIONS)}: {video_path}")
        if video_path in seen_videos:
            raise ValueError(f"Video appears in more than one pair: {video_path}")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Pair video not found: {video_path}")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Pair CSV not found: {csv_path}")
        seen_videos.add(video_path)
        pairs.append((video_path, csv_path))
    return tuple(sorted(pairs)) if pairs else None


def _strip_known_video_extension(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    return name[:-len(ext)] if ext in VIDEO_EXTENSIONS else name


def _parse_video_file(name: str) -> Optional[tuple[str, str]]:
    lower_name = name.lower()
    if lower_name.endswith(".remux.mp4"):
        return _strip_known_video_extension(name[:-len(".remux.mp4")]), "remux"
    if lower_name.endswith(".h264.mp4"):
        return _strip_known_video_extension(name[:-len(".h264.mp4")]), "h264"
    ext = os.path.splitext(name)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return None
    return name[:-len(ext)], "source"


def _find_video_csv_pairs(dataset_path: str):
    entries = sorted(os.listdir(dataset_path))
    all_files = {name for name in entries if os.path.isfile(os.path.join(dataset_path, name))}
    source_videos = []
    for index, name in enumerate(entries):
        parsed = _parse_video_file(name)
        if not parsed:
            continue
        stem, variant = parsed
        if variant == "source":
            source_videos.append((stem, os.path.join(dataset_path, name), index))

    video_stems = sorted({stem for stem, _, _ in source_videos}, key=len, reverse=True)
    csvs_by_stem: dict[str, list[tuple[str, int]]] = {}
    for index, name in enumerate(entries):
        if name not in all_files or not name.lower().endswith(".csv"):
            continue
        csv_stem = name[:-4]
        matched_stem = next(
            (stem for stem in video_stems if csv_stem == stem or csv_stem.startswith(f"{stem}_")),
            csv_stem,
        )
        csvs_by_stem.setdefault(matched_stem, []).append((name, index))

    pairs = []
    for stem, video_path, _ in source_videos:
        csv_candidates = csvs_by_stem.get(stem, [])
        if not csv_candidates:
            continue
        canonical = f"{stem}.csv"
        csv_candidates = sorted(csv_candidates, key=lambda item: (item[0] != canonical, item[1], item[0]))
        pairs.append((video_path, os.path.join(dataset_path, csv_candidates[0][0])))
    return pairs


def _parse_rational(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        parsed = float(text)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _probe_video(video_path: str) -> dict[str, Any]:
    """Return video size/duration/frame metadata with graceful fallbacks."""
    stat_size = os.path.getsize(video_path)
    metadata: dict[str, Any] = {
        "path": video_path,
        "size_bytes": stat_size,
        "duration_sec": 0.0,
        "fps": 30.0,
        "frame_count": 0,
        "width": 0,
        "height": 0,
    }

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate,duration:format=duration,size",
                "-of",
                "json",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(result.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}

        fps = _parse_rational(stream.get("avg_frame_rate")) or _parse_rational(stream.get("r_frame_rate"))
        duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
        nb_frames_raw = stream.get("nb_frames")
        frame_count = int(nb_frames_raw) if str(nb_frames_raw or "").isdigit() else 0
        if frame_count <= 0 and fps and duration > 0:
            frame_count = int(round(duration * fps))

        metadata.update(
            size_bytes=int(fmt.get("size") or stat_size),
            duration_sec=max(0.0, duration),
            fps=float(fps or 30.0),
            frame_count=max(0, frame_count),
            width=int(stream.get("width") or 0),
            height=int(stream.get("height") or 0),
        )
    except Exception:
        pass

    return metadata


def _read_behavior_counts_and_windows(
    csv_path: str,
    *,
    fps: float,
    clip_frames: int,
) -> tuple[Counter[str], set[int]]:
    counts: Counter[str] = Counter()
    windows: set[int] = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader: Iterable[dict[str, str]] = _csv_dict_reader(f)
        for row in reader:
            label = (row.get("labelId") or row.get("label") or "").strip()
            if label:
                counts[label] += 1

            try:
                start = float(row.get("timestamp", ""))
                end = float(row.get("endTimestamp", ""))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(start) or not math.isfinite(end):
                continue
            if end < start:
                start, end = end, start
            start_frame = max(0, int(math.floor(start * fps)))
            end_frame = max(start_frame, int(math.ceil(end * fps)))
            first_window = start_frame // clip_frames
            last_window = max(first_window, end_frame // clip_frames)
            for window in range(first_window, last_window + 1):
                windows.add(window)
    return counts, windows


def _default_work_dir(explicit_pairs: Optional[list[str]], work_dir: Optional[str]) -> str:
    if work_dir:
        return os.path.abspath(work_dir)
    if explicit_pairs:
        first = str(explicit_pairs[0]).split("=", 1)[0]
        if os.path.isabs(first):
            return os.path.dirname(first)
    raise ValueError("work_dir is required when pair paths are relative")


def _round_mb(value: float) -> int:
    return int(math.ceil(max(0.0, value)))


def _format_mb(value: int) -> str:
    if value >= 1024:
        return f"{value / 1024:.1f} GB"
    return f"{value} MB"


def _estimate_cached_video_disk_mb(annotated_frames: int, cache_resolution: int) -> int:
    if annotated_frames <= 0:
        return 0
    # H.264 CRF-23 144p animal-behavior clips generally land well below raw
    # RGB. 0.08 bytes/pixel is deliberately conservative for UI planning.
    encoded_bytes = annotated_frames * cache_resolution * cache_resolution * 0.08
    return _round_mb((encoded_bytes + 512 * 1024) / BYTES_PER_MB)


def _estimate_train_peak_vram_mb(model_size: str, resolution: int, behavior_kinds: int) -> int:
    """Estimate a one-step training peak for batch_size=1.

    The curves were calibrated from real trace-dev one-step train runs on
    2026-05-08. Small: 112p=1070MB, 144p=1529MB, 192p=2430MB,
    224p=3177MB. Large: 112p=5376MB, 144p=7147MB, 192p=10619MB,
    224p=13497MB.
    """
    class_mb = max(1, min(behavior_kinds, 50)) * (2 if model_size == "large" else 1)
    pixels = resolution * resolution
    if model_size == "large":
        return _round_mb(2666 + pixels * 0.216 + class_mb)
    return _round_mb(367 + pixels * 0.056 + class_mb)


def estimate_training_resources(
    *,
    work_dir: Optional[str] = None,
    explicit_pairs: Optional[list[str]] = None,
    clip_frames: int = 768,
    cache_resolution: int = 144,
) -> dict[str, Any]:
    """Estimate prep/training resources for selected video+CSV pairs."""
    clip_frames = max(1, int(clip_frames or 768))
    cache_resolution = max(16, int(cache_resolution or 144))
    dataset_path = _default_work_dir(explicit_pairs, work_dir)

    if explicit_pairs:
        pairs = list(_normalise_explicit_pairs(dataset_path, explicit_pairs) or [])
    else:
        pairs = _find_video_csv_pairs(dataset_path)
    if not pairs:
        raise FileNotFoundError("No video+CSV pairs found for resource estimation")

    behavior_counts: Counter[str] = Counter()
    video_summaries: list[dict[str, Any]] = []
    total_source_bytes = 0
    total_duration_sec = 0.0
    total_frames = 0
    total_windows = 0
    annotated_clip_count = 0
    annotated_frames = 0
    max_width = 0
    max_height = 0

    for video_path, csv_path in pairs:
        video_meta = _probe_video(video_path)
        fps = float(video_meta.get("fps") or 30.0)
        frame_count = int(video_meta.get("frame_count") or 0)
        duration_sec = float(video_meta.get("duration_sec") or 0.0)
        if frame_count <= 0 and duration_sec > 0:
            frame_count = int(round(duration_sec * fps))
        if duration_sec <= 0 and frame_count > 0 and fps > 0:
            duration_sec = frame_count / fps

        counts, windows = _read_behavior_counts_and_windows(
            csv_path,
            fps=fps,
            clip_frames=clip_frames,
        )
        behavior_counts.update(counts)

        video_windows = int(math.ceil(frame_count / clip_frames)) if frame_count > 0 else len(windows)
        source_bytes = int(video_meta.get("size_bytes") or 0)
        selected_windows = len(windows)
        selected_frames = sum(
            min(clip_frames, max(0, frame_count - (window * clip_frames)))
            for window in windows
        ) if frame_count > 0 else selected_windows * clip_frames

        total_source_bytes += source_bytes
        total_duration_sec += duration_sec
        total_frames += frame_count
        total_windows += video_windows
        annotated_clip_count += selected_windows
        annotated_frames += selected_frames
        max_width = max(max_width, int(video_meta.get("width") or 0))
        max_height = max(max_height, int(video_meta.get("height") or 0))
        video_summaries.append(
            {
                "name": Path(video_path).name,
                "csv": Path(csv_path).name,
                "source_mb": _round_mb(source_bytes / BYTES_PER_MB),
                "duration_sec": round(duration_sec, 2),
                "frames": frame_count,
                "annotated_clips": selected_windows,
                "annotations": sum(counts.values()),
                "width": int(video_meta.get("width") or 0),
                "height": int(video_meta.get("height") or 0),
            }
        )

    behavior_total = sum(behavior_counts.values())
    behavior_kinds = len(behavior_counts)
    coverage = annotated_clip_count / max(1, total_windows)

    virtual_disk_mb = _round_mb((total_frames * 8 + len(pairs) * 4096) / BYTES_PER_MB)
    cached_disk_mb = _estimate_cached_video_disk_mb(annotated_frames, cache_resolution)
    virtual_ram_mb = _round_mb(900 + min(total_source_bytes / BYTES_PER_GB, 32) * 120)
    cached_ram_mb = _round_mb(650 + max(1, annotated_clip_count) * 3)

    large_source = (
        total_source_bytes >= 4 * BYTES_PER_GB
        or total_duration_sec >= 3600
        or (max_width * max_height) >= (1280 * 720)
    )
    dense_annotations = coverage >= 0.6 and total_windows > 0
    if large_source and not dense_annotations and annotated_clip_count > 0:
        recommended_cache = "cached_video"
        cache_reason = "Large source videos with sparse annotated windows train faster from cached 144p clips."
    else:
        recommended_cache = "virtual"
        cache_reason = "The selected videos are compact enough that virtual clips avoid extra files."

    class_factor = max(1, min(behavior_kinds, 50))
    # Model-card VRAM is the startup footprint (model weights + EMA copy),
    # not a full training peak. Activations/optimizer states scale with the
    # selected resolution and are represented in the profile/cache estimates.
    small_vram_mb = _round_mb(160 + class_factor)
    large_vram_mb = _round_mb(1400 + class_factor * 3)
    small_ram_mb = _round_mb(1200 + behavior_total * 0.2 + len(pairs) * 8)
    large_ram_mb = _round_mb(2200 + behavior_total * 0.3 + len(pairs) * 12)
    recommended_model = (
        "large"
        if behavior_total >= 120 and behavior_kinds >= 3 and annotated_clip_count >= 40
        else "small"
    )
    resolution_options = [
        {
            "id": resolution,
            "label": f"{resolution}x{resolution}",
            "small_vram_mb": _estimate_train_peak_vram_mb("small", resolution, behavior_kinds),
            "large_vram_mb": _estimate_train_peak_vram_mb("large", resolution, behavior_kinds),
            "recommended": resolution == 144,
        }
        for resolution in TRAIN_RESOLUTIONS
    ]
    for option in resolution_options:
        option["detail"] = (
            f"train peak small ~{_format_mb(option['small_vram_mb'])}, "
            f"large ~{_format_mb(option['large_vram_mb'])}"
        )

    decoded_sample_mb = max(
        32.0,
        clip_frames * cache_resolution * cache_resolution * 3 * 2 / BYTES_PER_MB,
    )
    profile_estimates = []
    for profile in TRAIN_RESOURCE_PROFILES:
        ram_mb = _round_mb(
            1200
            + profile["num_workers"] * profile["prefetch_factor"] * decoded_sample_mb * 1.25
            + profile["decode_threads"] * 90
            + min(annotated_clip_count, 1000) * 1.5
        )
        profile_estimates.append(
            {
                **profile,
                "ram_mb": ram_mb,
                "vram_mb": 0,
                "detail": f"~{_format_mb(ram_mb)} RAM, no extra VRAM",
            }
        )

    if annotated_clip_count >= 250 or total_source_bytes >= 12 * BYTES_PER_GB:
        recommended_profile = "high"
    elif annotated_clip_count >= 40 or total_source_bytes >= 2 * BYTES_PER_GB:
        recommended_profile = "balanced"
    else:
        recommended_profile = "low"
    for profile in profile_estimates:
        profile["recommended"] = profile["id"] == recommended_profile

    cache_options = [
        {
            "id": "cached_video",
            "label": "Cached Video",
            "disk_mb": cached_disk_mb,
            "ram_mb": cached_ram_mb,
            "vram_mb": 0,
            "recommended": recommended_cache == "cached_video",
            "detail": f"~{_format_mb(cached_disk_mb)} disk, ~{_format_mb(cached_ram_mb)} RAM",
        },
        {
            "id": "virtual",
            "label": "Virtual",
            "disk_mb": virtual_disk_mb,
            "ram_mb": virtual_ram_mb,
            "vram_mb": 0,
            "recommended": recommended_cache == "virtual",
            "detail": f"~{_format_mb(virtual_disk_mb)} disk, ~{_format_mb(virtual_ram_mb)} RAM",
        },
    ]

    return {
        "summary": {
            "pair_count": len(pairs),
            "source_mb": _round_mb(total_source_bytes / BYTES_PER_MB),
            "duration_sec": round(total_duration_sec, 2),
            "frames": total_frames,
            "clip_frames": clip_frames,
            "total_windows": total_windows,
            "annotated_clip_count": annotated_clip_count,
            "annotation_count": behavior_total,
            "behavior_count": behavior_kinds,
            "coverage": round(coverage, 4),
            "max_width": max_width,
            "max_height": max_height,
            "cache_resolution": cache_resolution,
        },
        "videos": video_summaries,
        "behaviors": [
            {"name": name, "count": count}
            for name, count in sorted(behavior_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "cache_options": cache_options,
        "model_options": [
            {
                "id": "small",
                "label": "Small",
                "config_name": "tridet_small",
                "ram_mb": small_ram_mb,
                "vram_mb": small_vram_mb,
                "recommended": recommended_model == "small",
                "detail": f"load ~{_format_mb(small_vram_mb)} VRAM, ~{_format_mb(small_ram_mb)} RAM",
            },
            {
                "id": "large",
                "label": "Large",
                "config_name": "tridet_large",
                "ram_mb": large_ram_mb,
                "vram_mb": large_vram_mb,
                "recommended": recommended_model == "large",
                "detail": f"load ~{_format_mb(large_vram_mb)} VRAM, ~{_format_mb(large_ram_mb)} RAM",
            },
        ],
        "resolution_options": resolution_options,
        "resource_profiles": profile_estimates,
        "recommendations": {
            "cache_mode": recommended_cache,
            "cache_reason": cache_reason,
            "model_size": recommended_model,
            "resource_profile": recommended_profile,
            "notes": [
                "Resolution VRAM estimates show one-step training peaks for batch_size=1.",
                "Resource profile estimates cover dataloader RAM; no benchmark tuner is run in the UI flow.",
            ],
        },
    }
