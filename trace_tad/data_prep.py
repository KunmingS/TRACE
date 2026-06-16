"""Dataset auto-preparation: scan for videos+CSVs and generate annotations.

Given a dataset directory containing raw videos and per-video CSV annotations,
this module:
1. Writes dataset.json into a model artifact directory
2. Extracts class names from CSVs → classmap.txt
3. Records fixed-length training segments
4. Generates dataset.json in TRACE annotation format
"""

import csv
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import shutil
import subprocess

import cv2
import numpy as np

from trace_tad.model_artifacts import create_model_dir


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _resolve_cache_workers(cache_workers=None):
    if cache_workers is None:
        cpu = os.cpu_count() or 1
        return max(1, min(4, cpu // 4 if cpu >= 4 else 1))
    try:
        parsed = int(cache_workers)
    except (TypeError, ValueError):
        return 1
    return max(1, min(8, parsed))


def _emit_cache_log(logger, message, *, warning=False):
    if logger is not None:
        if warning:
            logger.warning(message)
        else:
            logger.info(message)
    else:
        level = "WARNING: " if warning else ""
        print(f"    {level}{message}")


def _cache_progress_interval(total):
    return max(1, min(25, total // 20 if total >= 20 else 1))


def _source_signature(video_path):
    stat = os.stat(video_path)
    return {
        "source_path": os.path.abspath(video_path),
        "source_mtime_ns": stat.st_mtime_ns,
        "source_size": stat.st_size,
    }


def _clip_cache_dir(video_path, clip_frames, cache_resolution, cache_crf):
    video_path = os.path.abspath(video_path)
    folder_name = f"{os.path.basename(video_path)}.trace-clips"
    cache_key = f"f{int(clip_frames)}_r{int(cache_resolution)}_crf{int(cache_crf)}"
    return os.path.join(os.path.dirname(video_path), folder_name, cache_key)


def _clip_cache_manifest_path(cache_dir):
    return os.path.join(cache_dir, "manifest.json")


def _clip_cache_manifest(video_path, clip_frames, cache_resolution, cache_crf):
    manifest = _source_signature(video_path)
    manifest.update({
        "clip_frames": int(clip_frames),
        "cache_resolution": int(cache_resolution),
        "cache_crf": int(cache_crf),
    })
    return manifest


def _write_clip_cache_manifest(cache_dir, video_path, clip_frames, cache_resolution, cache_crf):
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(_clip_cache_manifest_path(cache_dir), "w", encoding="utf-8") as f:
            json.dump(
                _clip_cache_manifest(video_path, clip_frames, cache_resolution, cache_crf),
                f,
                indent=2,
            )
    except OSError:
        pass


def _valid_clip_cache(cache_dir, video_path, clip_frames, cache_resolution, cache_crf, clip_names):
    try:
        with open(_clip_cache_manifest_path(cache_dir), "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    expected = _clip_cache_manifest(video_path, clip_frames, cache_resolution, cache_crf)
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False
    return all(os.path.isfile(os.path.join(cache_dir, name)) for name in clip_names)


def _extract_cached_clip_jobs(jobs, cache_workers=None, logger=None, label="clips"):
    if not jobs:
        return {}

    # The OpenCV fallback reads from one source in-process; keep that path
    # serial. ffmpeg subprocesses are safe to fan out, and each is capped to a
    # single encoder thread below.
    ffmpeg_available = shutil.which("ffmpeg") is not None
    workers = _resolve_cache_workers(cache_workers) if ffmpeg_available else 1
    workers = min(workers, len(jobs))
    _emit_cache_log(logger, f"Caching {len(jobs)} {label} with {workers} worker(s)")

    results = {}
    completed = 0
    cached_count = 0
    report_every = _cache_progress_interval(len(jobs))

    def run(job):
        try:
            ok, error = _extract_cached_video_clip(
                job["video_path"],
                job["clip_path"],
                job["start_time"],
                job["start_frame"],
                job["actual_frames"],
                job["avg_fps"],
                job["width"],
                job["height"],
                cache_resolution=job["cache_resolution"],
                cache_crf=job["cache_crf"],
            )
        except Exception as exc:
            ok, error = False, str(exc)
        return job["key"], ok, error

    def record(result):
        nonlocal completed, cached_count
        key, ok, error = result
        results[key] = (ok, error)
        completed += 1
        if ok:
            cached_count += 1
        if completed == 1 or completed == len(jobs) or completed % report_every == 0:
            _emit_cache_log(
                logger,
                f"Cached {completed}/{len(jobs)} {label} ({cached_count} written)",
            )

    if workers == 1:
        for job in jobs:
            record(run(job))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run, job) for job in jobs]
            for future in as_completed(futures):
                record(future.result())
    return results


def _pts_cache_path(video_path):
    """Path of the cached per-frame PTS table for a source video."""
    return os.fspath(video_path) + ".pts.npy"


def _pts_meta_path(video_path):
    """Sidecar carrying the source's mtime+size at the moment the PTS
    cache was built. Lets us validate the cache from `os.stat` alone,
    without re-opening the container."""
    return os.fspath(video_path) + ".pts.meta.json"


def _load_or_build_pts(video_path):
    """Return per-frame PTS array (seconds, ``float64``) for ``video_path``.

    The array carries one entry per encoded frame: ``pts[i]`` is the
    presentation timestamp of frame ``i`` as recorded in the container. This
    is the canonical source of truth for time ↔ frame conversion and works
    correctly for both CFR and VFR sources (USB webcams in dim labs are
    essentially always VFR — see ``docs/pts-based-frame-mapping.md``).
    Building it via decord reads only the index (no decoding), so it costs
    seconds even for hours of footage.

    The result is cached to ``<video_path>.pts.npy`` next to the source,
    paired with a ``<video_path>.pts.meta.json`` sidecar that records the
    source's mtime (ns) and size at build time.

    Cache invalidation: the cache is rebuilt if either file is missing OR
    the sidecar's recorded mtime/size doesn't match the source's current
    ``os.stat``. We deliberately do NOT re-open the container to length-
    check on every load — for a multi-GB MKV, decord's first open scans
    the whole container index and takes ~10 s, which would block every
    Editor video-open through the staged loading overlay. mtime+size is a
    strong-enough signal in practice; the pathological "overwrite with
    identical size and preserved mtime" case can still be forced via
    deleting the cache files.
    """
    import decord  # local import — keeps top-level import surface minimal

    cache_path = _pts_cache_path(video_path)
    meta_path = _pts_meta_path(video_path)
    src_stat = os.stat(video_path)

    if os.path.isfile(cache_path) and os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if (meta.get('sourceMtimeNs') == src_stat.st_mtime_ns
                    and meta.get('sourceSize') == src_stat.st_size):
                return np.load(cache_path).astype(np.float64, copy=False)
        except (OSError, ValueError):
            pass  # corrupt sidecar → fall through to rebuild

    vr = decord.VideoReader(os.fspath(video_path), num_threads=1)
    n = len(vr)
    pts = np.asarray(vr.get_frame_timestamp(range(n)), dtype=np.float64)[:, 0]
    try:
        np.save(cache_path, pts)
        # Re-stat after writing the npy: the source could (legitimately)
        # have been touched while decord was scanning, and we want the
        # sidecar to reflect the version we actually indexed.
        post_stat = os.stat(video_path)
        with open(meta_path, 'w') as f:
            json.dump({
                'sourceMtimeNs': post_stat.st_mtime_ns,
                'sourceSize': post_stat.st_size,
            }, f)
    except OSError:
        # Best-effort caching; fall back to the in-memory array if the
        # dataset directory is read-only.
        pass
    return pts


def _strip_known_video_extension(name):
    ext = os.path.splitext(name)[1].lower()
    return name[:-len(ext)] if ext in VIDEO_EXTENSIONS else name


def _parse_video_file(name):
    """Return (stem, variant) using the same grouping rule as PathPicker."""
    lower_name = name.lower()
    if lower_name.endswith(".remux.mp4"):
        return _strip_known_video_extension(name[:-len(".remux.mp4")]), "remux"
    if lower_name.endswith(".h264.mp4"):
        return _strip_known_video_extension(name[:-len(".h264.mp4")]), "h264"

    ext = os.path.splitext(name)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return None
    return name[:-len(ext)], "source"


def _normalise_included_stems(included_stems):
    if not included_stems:
        return None
    stems = [str(stem).strip() for stem in included_stems if str(stem).strip()]
    if not stems:
        return None
    return tuple(sorted(dict.fromkeys(stems)))


def _resolve_dataset_file(dataset_path, file_path):
    return file_path if os.path.isabs(file_path) else os.path.join(dataset_path, file_path)


def _normalise_explicit_pairs(dataset_path, explicit_pairs):
    if not explicit_pairs:
        return None

    pairs = []
    seen_videos = set()
    for spec in explicit_pairs:
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            video_spec, csv_spec = spec
            spec_label = f"{video_spec}={csv_spec}"
        else:
            spec = str(spec).strip()
            if not spec:
                continue
            spec_label = spec
            if "=" not in spec:
                raise ValueError(
                    f"Invalid pair spec '{spec}'. Use VIDEO_PATH=CSV_PATH."
                )
            video_spec, csv_spec = (part.strip() for part in spec.split("=", 1))
        if not video_spec or not csv_spec:
            raise ValueError(
                f"Invalid pair spec '{spec_label}'. Use VIDEO_PATH=CSV_PATH."
            )
        video_path = os.path.abspath(_resolve_dataset_file(dataset_path, video_spec))
        csv_path = os.path.abspath(_resolve_dataset_file(dataset_path, csv_spec))
        if video_path in seen_videos:
            raise ValueError(f"Video appears in more than one pair: {video_path}")
        seen_videos.add(video_path)
        pairs.append((video_path, csv_path))

    return tuple(sorted(pairs)) if pairs else None


def _find_video_csv_pairs(dataset_path, included_stems=None, explicit_pairs=None):
    """Find all (video_path, csv_path) pairs in a directory.

    If ``explicit_pairs`` is provided, it must contain ``VIDEO_PATH=CSV_PATH``
    specs. Relative paths are resolved against ``dataset_path``.

    Otherwise, ``included_stems`` can filter auto-discovered pairs by the
    same grouping key produced by the annotator's pair picker.
    """
    dataset_path = os.path.abspath(dataset_path)
    explicit_pairs = _normalise_explicit_pairs(dataset_path, explicit_pairs)
    if explicit_pairs:
        for video_path, csv_path in explicit_pairs:
            ext = os.path.splitext(video_path)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                raise ValueError(
                    f"Pair video must use one of {sorted(VIDEO_EXTENSIONS)}: {video_path}"
                )
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Pair video not found: {video_path}")
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"Pair CSV not found: {csv_path}")
        return list(explicit_pairs)

    included_stems = _normalise_included_stems(included_stems)
    allowlist = set(included_stems) if included_stems else None
    entries = sorted(os.listdir(dataset_path))

    source_videos = []
    for index, fname in enumerate(entries):
        parsed = _parse_video_file(fname)
        if not parsed:
            continue
        stem, variant = parsed
        if variant != "source":
            continue
        source_videos.append((stem, os.path.join(dataset_path, fname), index))

    video_stems = sorted({stem for stem, _, _ in source_videos}, key=len, reverse=True)
    csvs_by_stem = {}
    for index, fname in enumerate(entries):
        if not fname.lower().endswith(".csv"):
            continue
        csv_stem = fname[:-4]
        matched_stem = next(
            (stem for stem in video_stems if csv_stem == stem or csv_stem.startswith(f"{stem}_")),
            csv_stem,
        )
        csvs_by_stem.setdefault(matched_stem, []).append((fname, index))

    pairs = []
    for stem, video_path, _ in source_videos:
        if allowlist is not None and stem not in allowlist:
            continue
        csv_candidates = csvs_by_stem.get(stem, [])
        if not csv_candidates:
            continue
        canonical = f"{stem}.csv"
        csv_candidates = sorted(
            csv_candidates,
            key=lambda item: (item[0] != canonical, item[1], item[0]),
        )
        pairs.append((video_path, os.path.join(dataset_path, csv_candidates[0][0])))
    return pairs


def _csv_dict_reader(file_obj):
    """DictReader that skips `# trace-meta:` and other `#`-prefixed comment
    lines emitted by the annotator above the real header row.
    """
    return csv.DictReader(line for line in file_obj if not line.lstrip().startswith("#"))


def _extract_classes_from_csvs(csv_paths):
    """Collect all unique labels from CSV files, return sorted list."""
    labels = set()
    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = _csv_dict_reader(f)
            for row in reader:
                label = row["labelId"].strip()
                if label:
                    labels.add(label)
    return sorted(labels)


def _extract_cached_video_clip(
    video_path,
    clip_path,
    start_time,
    start_frame,
    actual_frames,
    avg_fps,
    width,
    height,
    cache_resolution=144,
    cache_crf=23,
):
    """Write a small training clip cache for one annotated window."""
    os.makedirs(os.path.dirname(clip_path), exist_ok=True)
    use_ffmpeg = shutil.which("ffmpeg") is not None

    if use_ffmpeg:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.4f}",
            "-i", video_path,
            "-frames:v", str(actual_frames),
            "-vf", f"scale={cache_resolution}:{cache_resolution}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", str(cache_crf),
            "-pix_fmt", "yuv420p",
            "-threads", "1",
            "-an",
            "-loglevel", "error",
            clip_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.isfile(clip_path):
            return True, None
        return False, result.stderr.strip()

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        clip_path,
        fourcc,
        avg_fps,
        (int(cache_resolution), int(cache_resolution)),
    )
    if not out.isOpened():
        cap.release()
        return False, f"Could not open VideoWriter for {clip_path}"

    written = 0
    for _ in range(actual_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (int(cache_resolution), int(cache_resolution)))
        out.write(frame)
        written += 1
    out.release()
    cap.release()
    if written == 0:
        return False, f"No frames written for {clip_path}"
    return True, None


def materialize_video_clips(
    video_path,
    output_dir,
    clip_frames=768,
    cache_resolution=144,
    cache_crf=23,
    cache_workers=None,
    clip_stem=None,
    logger=None,
):
    """Write resized cached clips for every window in a source video.

    This is the inference/test counterpart to the annotated training cache:
    it records source metadata for timeline conversion, but decodes from small
    clip files so long raw videos do not stay hot in DataLoader workers.
    """
    video_path = os.path.abspath(video_path)
    video_name = clip_stem or Path(video_path).stem
    pts_array = _load_or_build_pts(video_path)
    total_frames = len(pts_array)
    if total_frames <= 0:
        return []

    cap = cv2.VideoCapture(video_path)
    avg_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if avg_fps is None or avg_fps <= 0:
        span = float(pts_array[-1] - pts_array[0]) if total_frames > 1 else 0.0
        avg_fps = (total_frames - 1) / span if span > 0 else 30.0

    cache_dir = _clip_cache_dir(video_path, clip_frames, cache_resolution, cache_crf)
    pts_cache_path = _pts_cache_path(video_path)
    abs_pts_path = (
        os.path.abspath(pts_cache_path) if os.path.isfile(pts_cache_path) else None
    )
    clips = []
    jobs = []
    num_clips = (total_frames + clip_frames - 1) // clip_frames
    expected_clip_names = [
        f"{video_name}_clip_{clip_idx}.mp4"
        for clip_idx in range(num_clips)
    ]
    reuse_cache = _valid_clip_cache(
        cache_dir,
        video_path,
        clip_frames,
        cache_resolution,
        cache_crf,
        expected_clip_names,
    )
    if reuse_cache:
        _emit_cache_log(
            logger,
            f"Reusing {len(expected_clip_names)} cached inference clip(s): {cache_dir}",
        )

    for clip_idx in range(num_clips):
        start_frame = clip_idx * clip_frames
        actual_frames = min(clip_frames, total_frames - start_frame)
        if actual_frames <= 0:
            continue
        end_frame = start_frame + actual_frames - 1
        source_start_seconds = float(pts_array[start_frame] - pts_array[0])
        clip_duration = (
            float(pts_array[end_frame] - pts_array[start_frame]) + (1.0 / avg_fps)
        )
        clip_name = f"{video_name}_clip_{clip_idx}.mp4"
        clip_path = os.path.join(cache_dir, clip_name)

        clip_info = {
            "clip_idx": clip_idx,
            "frame": actual_frames,
            "duration": clip_duration,
            "source_video": video_path,
            "source_frame_offset": start_frame,
            "source_start_seconds": source_start_seconds,
        }
        if abs_pts_path is not None:
            clip_info["source_pts_table"] = abs_pts_path
        clips.append(clip_info)
        if not reuse_cache:
            jobs.append({
                "key": clip_idx,
                "video_path": video_path,
                "clip_path": clip_path,
                "clip_name": clip_name,
                "start_time": float(pts_array[start_frame]),
                "start_frame": start_frame,
                "actual_frames": actual_frames,
                "avg_fps": avg_fps,
                "width": width,
                "height": height,
                "cache_resolution": cache_resolution,
                "cache_crf": cache_crf,
            })

    results = _extract_cached_clip_jobs(
        jobs,
        cache_workers=cache_workers,
        logger=logger,
        label="inference clip(s)",
    )
    if jobs and sum(1 for ok, _ in results.values() if ok) == len(jobs):
        _write_clip_cache_manifest(
            cache_dir,
            video_path,
            clip_frames,
            cache_resolution,
            cache_crf,
        )
    for clip_info in clips:
        clip_idx = clip_info["clip_idx"]
        clip_path = os.path.join(cache_dir, f"{video_name}_clip_{clip_idx}.mp4")
        if reuse_cache:
            clip_info["cached_video"] = os.path.abspath(clip_path)
            continue
        ok, error = results.get(clip_idx, (False, "cache job did not run"))
        if ok:
            clip_info["cached_video"] = os.path.abspath(clip_path)
        else:
            _emit_cache_log(
                logger,
                f"Cached clip failed for {video_name}_clip_{clip_idx}.mp4: {error}; falling back to source decode",
                warning=True,
            )

    return clips


def _process_video(
    video_path,
    csv_path,
    output_dir,
    clip_frames=768,
    virtual_clips=True,
    cache_mode=None,
    cache_resolution=144,
    cache_crf=23,
    cache_workers=None,
):
    """Process a single video: map CSV times to frames, extract clip metadata.

    When `virtual_clips` is True (the default), no clip files are written:
    each clip entry records `source_video` and `source_frame_offset` so the
    training loader can read frames directly from the original video. This
    eliminates the CRF-18 re-encode step (zero quality loss, ~10x faster prep,
    and avoids duplicating gigabytes of clip data on disk).

    When `cache_mode` is "cached_video", only annotated windows are written as
    small resized MP4 clips for the training loader. Source metadata is still
    recorded for time mapping and debugging.

    When False, clips are physically extracted with ffmpeg as before.

    Returns dict of {clip_idx: {duration, frame, annotations, ...}} for clips
    that contain at least one annotation.
    """
    video_name = Path(video_path).stem
    print(f"  Processing video: {video_name}")
    if cache_mode is None:
        cache_mode = "virtual" if virtual_clips else "physical"

    # Load CSV annotations
    annotations = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _csv_dict_reader(f)
        for row in reader:
            annotations.append({
                "labelId": row["labelId"].strip(),
                "timestamp": float(row["timestamp"]),
                "endTimestamp": float(row["endTimestamp"]),
            })
    print(f"    {len(annotations)} annotations")

    # Build / load the PTS table — one canonical timestamp per encoded frame.
    # Replaces the previous cv2 CAP_PROP_POS_MSEC per-frame loop and is correct
    # for both CFR and VFR sources. See docs/pts-based-frame-mapping.md.
    print(f"    Loading PTS table...")
    pts_array = _load_or_build_pts(video_path)
    total_frames = len(pts_array)
    pts_cache_path = _pts_cache_path(video_path)

    # Resolution + average fps come from cv2 (display-only — not used for any
    # time ↔ frame conversion).
    cap = cv2.VideoCapture(video_path)
    avg_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if avg_fps is None or avg_fps <= 0:
        # Fallback: derive from PTS span when cv2 can't report it.
        span = float(pts_array[-1] - pts_array[0]) if total_frames > 1 else 0.0
        avg_fps = (total_frames - 1) / span if span > 0 else 30.0
    print(f"    {total_frames} frames, avg_fps={avg_fps:.3f}, {width}x{height}")

    # Map annotation times to frame indices via PTS searchsorted.
    for anno in annotations:
        s = int(np.searchsorted(pts_array, anno["timestamp"], side="left"))
        e = int(np.searchsorted(pts_array, anno["endTimestamp"], side="right") - 1)
        anno["start_frame"] = max(0, min(total_frames - 1, s))
        anno["end_frame"] = max(0, min(total_frames - 1, e))

    # Convert annotations to clip-relative format
    num_clips = (total_frames + clip_frames - 1) // clip_frames
    clips_data = {}

    for clip_idx in range(num_clips):
        clip_start = clip_idx * clip_frames
        clip_end = min((clip_idx + 1) * clip_frames - 1, total_frames - 1)
        clip_start_pts = float(pts_array[clip_start])

        clip_annos = []
        for anno in annotations:
            if anno["end_frame"] < clip_start or anno["start_frame"] > clip_end:
                continue
            rel_start = max(0, anno["start_frame"] - clip_start)
            rel_end = min(clip_frames - 1, anno["end_frame"] - clip_start)
            if rel_start <= rel_end:
                seg_start = float(pts_array[clip_start + rel_start]) - clip_start_pts
                seg_end = float(pts_array[clip_start + rel_end]) - clip_start_pts
                clip_annos.append({
                    "frame_segment": [rel_start, rel_end],
                    "segment": [seg_start, seg_end],
                    "timestamp_sec": [seg_start, seg_end],
                    "label": anno["labelId"],
                })

        if clip_annos:
            actual_frames = clip_end - clip_start + 1
            # Duration = inter-frame span across the clip + one frame's worth at
            # the local average fps (so a 30-frame clip @ 30fps reports 1.0s,
            # not 29/30s — matches the legacy `frame / fps` semantics).
            clip_duration = (
                float(pts_array[clip_end] - pts_array[clip_start]) + (1.0 / avg_fps)
            )
            clips_data[clip_idx] = {
                "frame": actual_frames,
                "duration": clip_duration,
                "annotations": clip_annos,
            }

    clips_with_annos = sorted(clips_data.keys())
    print(f"    {len(clips_with_annos)} clips with annotations")

    if not clips_with_annos:
        return clips_data

    abs_video_path = os.path.abspath(video_path)
    abs_pts_path = (
        os.path.abspath(pts_cache_path) if os.path.isfile(pts_cache_path) else None
    )

    if cache_mode == "virtual":
        # No physical extraction — record source video + frame offset + PTS
        # table for each clip. The PTS table reference lets training-time
        # loaders convert clip-local frame indices back to source-video
        # seconds without rebuilding the index.
        for clip_idx in clips_with_annos:
            start_frame = clip_idx * clip_frames
            clips_data[clip_idx]["source_video"] = abs_video_path
            clips_data[clip_idx]["source_frame_offset"] = start_frame
            if abs_pts_path is not None:
                clips_data[clip_idx]["source_pts_table"] = abs_pts_path
        print(f"    Recorded {len(clips_with_annos)} virtual clips (no re-encoding)")
    elif cache_mode == "cached_video":
        cache_dir = _clip_cache_dir(video_path, clip_frames, cache_resolution, cache_crf)
        jobs = []
        expected_clip_names = [
            f"{video_name}_clip_{clip_idx}.mp4"
            for clip_idx in clips_with_annos
        ]
        reuse_cache = _valid_clip_cache(
            cache_dir,
            video_path,
            clip_frames,
            cache_resolution,
            cache_crf,
            expected_clip_names,
        )
        if reuse_cache:
            print(f"    Reusing {len(expected_clip_names)} cached training videos: {cache_dir}")
        for clip_idx in clips_with_annos:
            start_frame = clip_idx * clip_frames
            actual_frames = min(clip_frames, total_frames - start_frame)
            start_time = float(pts_array[start_frame])
            clip_name = f"{video_name}_clip_{clip_idx}.mp4"
            clip_path = os.path.join(cache_dir, clip_name)
            clips_data[clip_idx]["source_video"] = abs_video_path
            clips_data[clip_idx]["source_frame_offset"] = start_frame
            if abs_pts_path is not None:
                clips_data[clip_idx]["source_pts_table"] = abs_pts_path
            if not reuse_cache:
                jobs.append({
                    "key": clip_idx,
                    "video_path": video_path,
                    "clip_path": clip_path,
                    "clip_name": clip_name,
                    "start_time": start_time,
                    "start_frame": start_frame,
                    "actual_frames": actual_frames,
                    "avg_fps": avg_fps,
                    "width": width,
                    "height": height,
                    "cache_resolution": cache_resolution,
                    "cache_crf": cache_crf,
                })
        results = _extract_cached_clip_jobs(
            jobs,
            cache_workers=cache_workers,
            label="annotated clip(s)",
        )
        if jobs and sum(1 for ok, _ in results.values() if ok) == len(jobs):
            _write_clip_cache_manifest(
                cache_dir,
                video_path,
                clip_frames,
                cache_resolution,
                cache_crf,
            )
        cached_count = 0
        for clip_idx in clips_with_annos:
            clip_path = os.path.join(cache_dir, f"{video_name}_clip_{clip_idx}.mp4")
            if reuse_cache:
                clips_data[clip_idx]["cached_video"] = os.path.abspath(clip_path)
                cached_count += 1
                continue
            ok, error = results.get(clip_idx, (False, "cache job did not run"))
            if not ok:
                print(
                    f"    WARNING: cached video failed for {video_name}_clip_{clip_idx}.mp4: "
                    f"{error}; falling back to virtual clip"
                )
                continue
            clips_data[clip_idx]["cached_video"] = os.path.abspath(clip_path)
            cached_count += 1
        print(f"    Wrote {cached_count} cached training videos ({cache_resolution}x{cache_resolution})")
    else:
        # Extract clip videos using ffmpeg (legacy path, kept for compatibility).
        os.makedirs(output_dir, exist_ok=True)
        use_ffmpeg = shutil.which("ffmpeg") is not None

        for clip_idx in clips_with_annos:
            clip_name = f"{video_name}_clip_{clip_idx}.mp4"
            clip_path = os.path.join(output_dir, clip_name)
            start_frame = clip_idx * clip_frames
            actual_frames = min(clip_frames, total_frames - start_frame)
            # PTS-derived clip start / duration. ffmpeg's -ss + -t expect
            # source-video seconds; the same numbers used to come from
            # `start_frame / fps` and `actual_frames / fps`.
            start_time = float(pts_array[start_frame])
            last_frame = start_frame + actual_frames - 1
            duration_sec = (
                float(pts_array[last_frame] - pts_array[start_frame])
                + (1.0 / avg_fps)
            )

            if use_ffmpeg:
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{start_time:.4f}",
                    "-i", video_path,
                    "-t", f"{duration_sec:.4f}",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-an",
                    "-loglevel", "error",
                    clip_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"    WARNING: ffmpeg failed for {clip_name}: {result.stderr.strip()}")
                    continue
            else:
                # Fallback to OpenCV (mp4v codec). The output container is CFR
                # at avg_fps — fine for the legacy --reencode-clips workflow,
                # since virtual clips are now the default and this branch is
                # only reached when ffmpeg is unavailable.
                cap2 = cv2.VideoCapture(video_path)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(clip_path, fourcc, avg_fps, (width, height))
                if not out.isOpened():
                    print(f"    WARNING: Could not open VideoWriter for {clip_name}, skipping")
                    cap2.release()
                    continue
                for _ in range(actual_frames):
                    ret, frame = cap2.read()
                    if not ret:
                        break
                    out.write(frame)
                out.release()
                cap2.release()

            print(f"    Created clip {clip_idx}: {clip_name} ({actual_frames} frames)")

    return clips_data


def _generate_classmap_from_json(json_path, classmap_path):
    """Extract sorted unique labels from a dataset JSON and write classmap.txt."""
    with open(json_path, "r", encoding="utf-8") as f:
        database = json.load(f)["database"]

    labels = set()
    for video_info in database.values():
        for anno in video_info.get("annotations", []):
            label = anno.get("label", "").strip()
            if label:
                labels.add(label)

    class_map = sorted(labels)
    with open(classmap_path, "w", encoding="utf-8") as f:
        for name in class_map:
            f.write(name + "\n")
    return class_map


def _dataset_has_cached_videos(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            database = json.load(f)["database"]
    except (OSError, KeyError, json.JSONDecodeError):
        return False
    if not database:
        return False
    for video_info in database.values():
        cached_video = video_info.get("cached_video")
        if not cached_video or not os.path.isfile(cached_video):
            return False
        if video_info.get("source_video"):
            return False
    return True


def materialize_dataset_cached_videos(
    annotation_path,
    output_dir,
    clip_frames=768,
    cache_resolution=144,
    cache_crf=23,
    cache_workers=None,
    logger=None,
):
    """Return a dataset JSON whose source-backed clips have cached videos.

    Existing cached entries are reused. Virtual/source-backed entries are
    materialized next to the source video and referenced from
    ``output_dir/dataset_cached.json``. Entries without ``source_video`` are
    left unchanged because they already point at legacy physical clips.
    """
    if _dataset_has_cached_videos(annotation_path):
        return annotation_path

    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    database = payload.get("database", {})
    output_dir = os.path.abspath(output_dir)
    source_cache = {}
    cache_groups = {}
    changed = False
    cached_count = 0
    reused_count = 0
    jobs = []

    for video_name, entry in database.items():
        cached_video = entry.get("cached_video")
        source_video = entry.get("source_video")
        if cached_video and os.path.isfile(cached_video) and not source_video:
            cached_count += 1
            continue
        if not source_video:
            continue

        source_video = os.path.abspath(source_video)
        if source_video not in source_cache:
            pts_array = _load_or_build_pts(source_video)
            cap = cv2.VideoCapture(source_video)
            avg_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if avg_fps is None or avg_fps <= 0:
                span = float(pts_array[-1] - pts_array[0]) if len(pts_array) > 1 else 0.0
                avg_fps = (len(pts_array) - 1) / span if span > 0 else 30.0
            source_cache[source_video] = (pts_array, avg_fps, width, height)

        pts_array, avg_fps, width, height = source_cache[source_video]
        start_frame = int(entry.get("source_frame_offset", 0))
        actual_frames = int(entry.get("frame", 0))
        if actual_frames <= 0 or start_frame >= len(pts_array):
            continue
        actual_frames = min(actual_frames, len(pts_array) - start_frame)
        cache_dir = _clip_cache_dir(source_video, clip_frames, cache_resolution, cache_crf)
        cache_groups[cache_dir] = source_video
        clip_name = f"{video_name}.mp4"
        clip_path = os.path.join(cache_dir, f"{video_name}.mp4")
        if _valid_clip_cache(
            cache_dir,
            source_video,
            clip_frames,
            cache_resolution,
            cache_crf,
            [clip_name],
        ):
            abs_clip_path = os.path.abspath(clip_path)
            if entry.get("cached_video") != abs_clip_path:
                entry["cached_video"] = abs_clip_path
                changed = True
            cached_count += 1
            reused_count += 1
            continue
        jobs.append({
            "key": video_name,
            "video_path": source_video,
            "clip_path": clip_path,
            "clip_name": clip_name,
            "start_time": float(pts_array[start_frame]),
            "start_frame": start_frame,
            "actual_frames": actual_frames,
            "avg_fps": avg_fps,
            "width": width,
            "height": height,
            "cache_resolution": cache_resolution,
            "cache_crf": cache_crf,
        })

    results = _extract_cached_clip_jobs(
        jobs,
        cache_workers=cache_workers,
        logger=logger,
        label="eval clip(s)",
    )
    successful_cache_dirs = set()
    for video_name, (ok, error) in results.items():
        entry = database[video_name]
        if ok:
            source_video = os.path.abspath(entry["source_video"])
            cache_dir = _clip_cache_dir(source_video, clip_frames, cache_resolution, cache_crf)
            clip_path = os.path.join(cache_dir, f"{video_name}.mp4")
            entry["cached_video"] = os.path.abspath(clip_path)
            changed = True
            cached_count += 1
            successful_cache_dirs.add(cache_dir)
        elif logger is not None:
            logger.warning(
                f"Cached eval clip failed for {video_name}: {error}; falling back to source decode"
            )
    for cache_dir in successful_cache_dirs:
        _write_clip_cache_manifest(
            cache_dir,
            cache_groups[cache_dir],
            clip_frames,
            cache_resolution,
            cache_crf,
        )

    if not changed:
        return annotation_path

    os.makedirs(output_dir, exist_ok=True)
    cached_path = os.path.join(output_dir, "dataset_cached.json")
    with open(cached_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if logger is not None:
        if reused_count:
            logger.info(f"Reused {reused_count} cached eval clip(s) from source sidecar cache")
        logger.info(f"Cached eval dataset clips: {cached_count}/{len(database)}")
        logger.info(f"Cached eval dataset JSON: {cached_path}")
    return cached_path


# Synthetic stratum used to keep background-only clips (no behavior labels)
# proportional across the train/val/test split alongside the real classes.
_BG_STRATUM = "(background)"


def _clip_label_set(clip_entry):
    """Behavior labels present in one clip (empty set => background-only)."""
    labels = set()
    for anno in clip_entry.get("annotations", []) or []:
        lab = anno.get("label")
        if lab is not None and str(lab).strip():
            labels.add(str(lab).strip())
    return labels


def _resolve_split_ratios(train_ratio, val_ratio, test_ratio):
    """Resolve a normalized ``(train, val, test)`` fraction triple.

    Back-compat: when only ``train_ratio`` is supplied (``val_ratio`` and
    ``test_ratio`` left as ``None``), the remaining mass is divided evenly
    between validation and test so legacy 2-way callers transparently gain a
    held-out test split. Passing ``test_ratio=0`` explicitly yields a 2-way
    train/val split.
    """
    train_ratio = float(train_ratio)
    if val_ratio is None and test_ratio is None:
        rem = max(0.0, 1.0 - train_ratio)
        val_ratio = test_ratio = rem / 2.0
    elif val_ratio is None:
        val_ratio = max(0.0, 1.0 - train_ratio - float(test_ratio))
    elif test_ratio is None:
        test_ratio = max(0.0, 1.0 - train_ratio - float(val_ratio))

    train_ratio = max(0.0, train_ratio)
    val_ratio = max(0.0, float(val_ratio))
    test_ratio = max(0.0, float(test_ratio))
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return train_ratio / total, val_ratio / total, test_ratio / total


def _stratified_split(clip_records, split_ratios, seed=42):
    """Distribute multi-label clips into named splits with even per-class shares.

    Implements the multi-label iterative stratification of Sechidis, Tsoumakas
    & Vlahavas (2011): repeatedly pick the label with the fewest remaining
    unassigned clips and hand each of those clips to whichever split has the
    largest remaining quota for that label (ties broken by largest overall
    remaining quota, then deterministically via a seeded RNG). This keeps every
    behavior category — and the background-only stratum — at a consistent
    proportion across train/val/test, which a per-video random cut does not.

    Args:
        clip_records: list of ``(clip_key, label_set)`` pairs. An empty
            ``label_set`` marks a background-only clip; it is stratified under a
            synthetic background label so those clips stay proportional too.
        split_ratios: list of ``(split_name, fraction)`` with fractions that
            sum to 1.
        seed: RNG seed for deterministic tie-breaking.

    Returns:
        dict mapping ``clip_key`` -> ``split_name``.
    """
    rng = random.Random(seed)
    names = [n for n, _ in split_ratios]
    fracs = {n: f for n, f in split_ratios}

    label_pool = defaultdict(set)   # label -> set of still-unassigned clip keys
    clip_labels = {}                # clip key -> set of strata it belongs to
    for clip_key, label_set in clip_records:
        strata = set(label_set) if label_set else {_BG_STRATUM}
        clip_labels[clip_key] = strata
        for lab in strata:
            label_pool[lab].add(clip_key)

    n_total = len(clip_records)
    desired_total = {n: fracs[n] * n_total for n in names}
    desired_label = {
        lab: {n: fracs[n] * len(keys) for n in names}
        for lab, keys in label_pool.items()
    }

    assignment = {}
    unassigned = set(clip_labels.keys())

    def _assign(clip_key, split_name):
        assignment[clip_key] = split_name
        unassigned.discard(clip_key)
        desired_total[split_name] -= 1
        for lab in clip_labels[clip_key]:
            if clip_key in label_pool[lab]:
                label_pool[lab].discard(clip_key)
                desired_label[lab][split_name] -= 1

    while unassigned:
        candidate_labels = [lab for lab, keys in label_pool.items() if keys]
        if not candidate_labels:
            break
        # Rarest remaining label first; deterministic tie-break by name.
        lab = min(candidate_labels, key=lambda l: (len(label_pool[l]), l))
        for clip_key in sorted(label_pool[lab]):
            if clip_key not in unassigned:
                continue
            best = max(
                names,
                key=lambda n: (desired_label[lab][n], desired_total[n], rng.random()),
            )
            _assign(clip_key, best)

    # Leftovers (clips whose every stratum pool emptied) -> by total quota.
    for clip_key in sorted(unassigned):
        best = max(names, key=lambda n: (desired_total[n], rng.random()))
        _assign(clip_key, best)

    return assignment


def _log_split_composition(database, class_map):
    """Print per-split clip counts and per-class clip counts for verification."""
    per_split = defaultdict(Counter)
    totals = Counter()
    for entry in database.values():
        s = entry["subset"]
        totals[s] += 1
        labs = _clip_label_set(entry)
        if not labs:
            per_split[s][_BG_STRATUM] += 1
        for lab in labs:
            per_split[s][lab] += 1

    order = [s for s in ("train", "validation", "test") if totals.get(s)]
    print("Split composition (stratified by behavior category):")
    print("  totals: " + ", ".join(f"{s}={totals[s]}" for s in order))
    for lab in list(class_map) + [_BG_STRATUM]:
        counts = ", ".join(f"{s}={per_split[s].get(lab, 0)}" for s in order)
        print(f"    {lab}: {counts}")


def prepare_dataset(dataset_path, clip_frames=768, train_ratio=0.7,
                    val_ratio=None, test_ratio=None, virtual_clips=True,
                    included_stems=None, explicit_pairs=None, output_dir=None,
                    cache_mode=None, cache_resolution=144, cache_crf=23,
                    cache_workers=None):
    """Prepare a dataset directory for training.

    Scans for video+CSV pairs and generates dataset.json and classmap.txt.
    Skips if the output directory already exists with dataset.json.

    Args:
        dataset_path: Directory containing videos and CSV annotations.
        clip_frames: Number of frames per clip (default: 768).
        train_ratio: Fraction of clips for training (default: 0.7).
        val_ratio: Fraction of clips for validation (best-epoch selection +
            threshold tuning). Defaults to half of the remainder after
            ``train_ratio`` when both ``val_ratio`` and ``test_ratio`` are None.
        test_ratio: Fraction of clips for the held-out test split (unbiased
            final reporting only — never used to pick the checkpoint or the
            threshold). Defaults to the other half of the remainder. Pass 0 for
            a 2-way train/val split. The split is stratified by behavior
            category so each class keeps a consistent proportion across splits.
        virtual_clips: If True (default), record `source_video` + `source_frame_offset`
            metadata instead of re-encoding clip files. Eliminates CRF-18 quality loss
            and ~10x speedup. Pass False for `--reencode-clips` workflows that want
            self-contained clip files (e.g. shipping a dataset to another machine).
        cache_mode: Optional explicit cache mode. ``"virtual"`` preserves the
            default virtual-clip behavior, while ``"cached_video"`` writes only
            annotated windows as resized MP4 clips and points ``dataset.json`` at
            those files for training.
        included_stems: Optional iterable of video stems to include when using
            automatic video+CSV discovery. Stems match the same grouping key
            the picker emits (filename minus extension, with .remux/.h264 copy
            suffixes collapsed).
        explicit_pairs: Optional iterable of ``VIDEO_PATH=CSV_PATH`` specs.
            Relative paths are resolved against ``dataset_path``. When set,
            auto-discovery is skipped and only these exact pairs are processed.
        output_dir: Optional directory to write ``dataset.json`` and
            ``classmap.txt``. When omitted, TRACE creates a new ``model_``
            timestamp directory under ``dataset_path``.

    Returns:
        (output_dir, json_path, classmap_path) tuple of paths.
    """
    dataset_path = os.path.abspath(dataset_path)
    if cache_mode is None:
        cache_mode = "virtual" if virtual_clips else "physical"
    if cache_mode not in ("virtual", "cached_video", "physical"):
        raise ValueError(f"Unsupported cache_mode: {cache_mode}")
    included_stems = _normalise_included_stems(included_stems)
    explicit_pairs = _normalise_explicit_pairs(dataset_path, explicit_pairs)
    if output_dir is None:
        output_dir = create_model_dir(dataset_path)
    else:
        output_dir = os.path.abspath(output_dir)
    json_path = os.path.join(output_dir, "dataset.json")
    classmap_path = os.path.join(output_dir, "classmap.txt")

    # Check if this output directory already has dataset metadata.
    if os.path.isdir(output_dir) and os.path.isfile(json_path):
        has_required_cache = cache_mode != "cached_video" or _dataset_has_cached_videos(json_path)
        # Ensure classmap exists — regenerate from JSON if missing
        if has_required_cache and not os.path.isfile(classmap_path):
            _generate_classmap_from_json(json_path, classmap_path)
            print(f"Generated classmap: {classmap_path}")
        if has_required_cache:
            print(f"Dataset already prepared at {output_dir}")
            return output_dir, json_path, classmap_path
        print(f"Dataset at {output_dir} is missing cached videos; rebuilding")

    # Find video-CSV pairs (optionally filtered by stem allowlist)
    pairs = _find_video_csv_pairs(
        dataset_path,
        included_stems=included_stems,
        explicit_pairs=explicit_pairs,
    )
    if not pairs:
        exts = ", ".join(sorted(VIDEO_EXTENSIONS))
        if included_stems:
            stems_str = ", ".join(sorted(set(included_stems)))
            raise FileNotFoundError(
                f"Selected stems matched no pairs in {dataset_path}. "
                f"Requested stems: {stems_str}. "
                f"Expected matching video ({exts}) and .csv files for each stem."
            )
        raise FileNotFoundError(
            f"No video+CSV pairs found in {dataset_path}. "
            f"Expected matching video ({exts}) and .csv files "
            "(e.g., video1.mp4 + video1.csv, or video1.mkv + video1.csv)."
        )
    print(f"Found {len(pairs)} video-CSV pairs")

    # Extract class map from all CSVs
    csv_paths = [csv_path for _, csv_path in pairs]
    class_map = _extract_classes_from_csvs(csv_paths)
    if not class_map:
        raise ValueError(f"No labels found in CSV files in {dataset_path}")
    print(f"Classes: {class_map}")

    # Process each video
    all_clips_data = {}
    for video_path, csv_path in pairs:
        video_name = Path(video_path).stem
        clips_data = _process_video(
            video_path,
            csv_path,
            output_dir,
            clip_frames,
            virtual_clips=virtual_clips,
            cache_mode=cache_mode,
            cache_resolution=cache_resolution,
            cache_crf=cache_crf,
            cache_workers=cache_workers,
        )
        all_clips_data[video_name] = clips_data

    # Generate dataset.json with a stratified train/val/test split.
    # The split is computed globally at clip level (not per-video) so behavior
    # categories stay proportional across splits. val drives best-epoch
    # selection + threshold tuning; test is held out for unbiased reporting.
    train_r, val_r, test_r = _resolve_split_ratios(train_ratio, val_ratio, test_ratio)
    clip_records = []
    for video_name, clips_data in all_clips_data.items():
        for clip_idx in clips_data.keys():
            clip_key = f"{video_name}_clip_{clip_idx}"
            clip_records.append((clip_key, _clip_label_set(clips_data[clip_idx])))

    split_ratios = [("train", train_r), ("validation", val_r)]
    if test_r > 0:
        split_ratios.append(("test", test_r))
    assignment = _stratified_split(clip_records, split_ratios, seed=42)

    database = {}
    for video_name, clips_data in all_clips_data.items():
        for clip_idx in clips_data.keys():
            clip_key = f"{video_name}_clip_{clip_idx}"
            subset = assignment.get(clip_key, "train")
            entry = {
                "duration": clips_data[clip_idx]["duration"],
                "frame": clips_data[clip_idx]["frame"],
                "subset": subset,
                "annotations": clips_data[clip_idx]["annotations"],
            }
            # Virtual-clip metadata (omitted for legacy physical clips)
            if "source_video" in clips_data[clip_idx]:
                entry["source_video"] = clips_data[clip_idx]["source_video"]
                entry["source_frame_offset"] = clips_data[clip_idx]["source_frame_offset"]
                # PTS table reference — present whenever the cache wrote
                # successfully. Loaders that don't know about it ignore it.
                if "source_pts_table" in clips_data[clip_idx]:
                    entry["source_pts_table"] = clips_data[clip_idx]["source_pts_table"]
            if "cached_video" in clips_data[clip_idx]:
                entry["cached_video"] = clips_data[clip_idx]["cached_video"]
            database[clip_key] = entry

    _log_split_composition(database, class_map)

    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"database": database}, f, indent=2, ensure_ascii=False)
    print(f"Dataset JSON saved: {json_path}")

    # Write classmap
    with open(classmap_path, "w", encoding="utf-8") as f:
        for name in class_map:
            f.write(name + "\n")
    print(f"Class map saved: {classmap_path}")

    return output_dir, json_path, classmap_path
