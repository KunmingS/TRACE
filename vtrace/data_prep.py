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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import shutil
import subprocess

import cv2
import numpy as np

from vtrace.model_artifacts import create_model_dir
from vtrace.proxy_geometry import ProxyGeometry


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Decode proxies: one downscaled, frame-aligned copy of each source video.
# Bump PROXY_VERSION whenever the encode recipe changes, so every existing
# proxy is treated as stale without touching the manifest schema.
PROXY_VERSION = 1
PROXY_GOP = 30
PROXY_CRF = 23
DEFAULT_PROXY_GEOMETRY = ProxyGeometry(144, True)

_NVENC_CACHE = None


def _resolve_proxy_workers(workers=None):
    if workers is None:
        cpu = os.cpu_count() or 1
        return max(1, min(4, cpu // 4 if cpu >= 4 else 1))
    try:
        parsed = int(workers)
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


def _proxy_cache_dir(video_path, geometry, crf):
    """Sidecar directory holding one decode proxy of ``video_path``.

    The key deliberately omits the clip/window length: a proxy is full-length,
    so changing ``window_size`` no longer invalidates it (the old per-window
    clip cache keyed on ``f{frames}`` and had to be rebuilt from scratch).
    ``_v{PROXY_VERSION}`` lets a change to the encode recipe invalidate every
    proxy without migrating the manifest schema.
    """
    video_path = os.path.abspath(video_path)
    folder_name = f"{os.path.basename(video_path)}.trace-proxy"
    cache_key = f"{geometry.key}_crf{int(crf)}_g{PROXY_GOP}_v{PROXY_VERSION}"
    return os.path.join(os.path.dirname(video_path), folder_name, cache_key)


def _proxy_path(video_path, geometry, crf):
    return os.path.join(_proxy_cache_dir(video_path, geometry, crf), "proxy.mp4")


def _proxy_manifest_path(cache_dir):
    return os.path.join(cache_dir, "manifest.json")


def _proxy_manifest(video_path, geometry, crf, source_frames, proxy_frames):
    manifest = _source_signature(video_path)
    manifest.update({
        "proxy_short_side": int(geometry.short_side),
        "proxy_square": bool(geometry.square),
        "proxy_crf": int(crf),
        "proxy_gop": PROXY_GOP,
        "proxy_version": PROXY_VERSION,
        "source_frame_count": int(source_frames),
        "proxy_frame_count": int(proxy_frames),
    })
    return manifest


def _write_proxy_manifest(cache_dir, video_path, geometry, crf, source_frames, proxy_frames):
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(_proxy_manifest_path(cache_dir), "w", encoding="utf-8") as f:
            json.dump(
                _proxy_manifest(video_path, geometry, crf, source_frames, proxy_frames),
                f,
                indent=2,
            )
    except OSError:
        pass


def _valid_proxy(video_path, geometry, crf, source_frames):
    """True when a reusable proxy already exists for this source + geometry."""
    cache_dir = _proxy_cache_dir(video_path, geometry, crf)
    proxy_path = _proxy_path(video_path, geometry, crf)
    if not os.path.isfile(proxy_path):
        return False
    try:
        with open(_proxy_manifest_path(cache_dir), "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    expected = _proxy_manifest(
        video_path, geometry, crf, source_frames, source_frames
    )
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False
    # The frame counts must agree with each other too: a proxy whose frame
    # count drifted from the source silently misaligns every annotation.
    if manifest.get("proxy_frame_count") != manifest.get("source_frame_count"):
        return False
    return os.path.getmtime(proxy_path) >= os.path.getmtime(video_path)


def _proxy_frame_count(proxy_path):
    """Frame count of a built proxy, or -1 if it cannot be read."""
    try:
        import decord

        return len(decord.VideoReader(os.fspath(proxy_path), num_threads=1))
    except Exception:
        return -1


def _nvenc_available():
    """Whether NVENC can actually encode here.

    Listing the encoder is not enough: ffmpeg advertises `h264_nvenc` from the
    build config, so a box with a driver/library mismatch or no GPU still
    reports it and then fails at encode time with "No capable devices found".
    Probe by encoding one synthetic frame.
    """
    global _NVENC_CACHE
    if _NVENC_CACHE is None:
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "color=black:s=64x64:d=1",
                 "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, text=True, timeout=30,
            )
            _NVENC_CACHE = result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _NVENC_CACHE = False
    return _NVENC_CACHE


def _proxy_ffmpeg_cmd(video_path, geometry, crf, out_path, use_nvenc=None):
    """ffmpeg args for a frame-aligned decode proxy.

    Three settings are load-bearing:

    * ``-fps_mode passthrough`` — the default (``cfr``) duplicates or drops
      frames on VFR input, which would destroy the 1:1 frame alignment the
      whole proxy design rests on. TRACE targets VFR webcam recordings.
    * ``setpts=N/(30*TB)`` — renumbers output PTS monotonically by frame index.
      VFR sources with duplicate timestamps otherwise make the mp4 muxer emit
      non-monotonic DTS. Safe because the proxy's own PTS are never read: the
      source's ``.pts.npy`` stays the timeline authority.
    * fixed short GOP — windows seek to arbitrary frames, and x264's default
      adaptive 250-frame GOP would make every seek decode up to 250 frames of
      slack, giving back most of the speedup.
    """
    cmd = [
        "ffmpeg", "-y", "-i", os.fspath(video_path),
        "-an", "-sn",
        "-vf", f"{geometry.vf}:flags=bicubic,setpts=N/(30*TB)",
        "-fps_mode", "passthrough",
    ]
    if _nvenc_available() if use_nvenc is None else use_nvenc:
        # CPU decode + CPU scale + GPU encode. Deliberately no
        # `-hwaccel_output_format cuda`, which would keep frames in VRAM and
        # force the scale_cuda/scale_npp filters instead of the CPU `scale`.
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(int(crf))]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf))]
    cmd += [
        "-pix_fmt", "yuv420p",
        "-g", str(PROXY_GOP), "-keyint_min", str(PROXY_GOP), "-sc_threshold", "0",
        "-movflags", "+faststart",
        "-loglevel", "error",
        os.fspath(out_path),
    ]
    return cmd


def build_video_proxy(video_path, geometry, crf=PROXY_CRF, source_frames=None, logger=None):
    """Build (or reuse) the decode proxy for one video.

    Returns the absolute proxy path, or None when no usable proxy exists — a
    missing proxy is never fatal, callers simply decode from the source.
    """
    video_path = os.path.abspath(video_path)
    if source_frames is None:
        source_frames = len(_load_or_build_pts(video_path))
    if source_frames <= 0:
        return None

    proxy_path = _proxy_path(video_path, geometry, crf)
    if _valid_proxy(video_path, geometry, crf, source_frames):
        return proxy_path

    if shutil.which("ffmpeg") is None:
        _emit_cache_log(
            logger,
            "ffmpeg not found; decoding from source video instead of a proxy",
            warning=True,
        )
        return None

    cache_dir = os.path.dirname(proxy_path)
    os.makedirs(cache_dir, exist_ok=True)
    # Insert .tmp before the extension so ffmpeg can still infer the muxer.
    tmp_path = os.path.join(cache_dir, "proxy.tmp.mp4")
    try:
        result = subprocess.run(
            _proxy_ffmpeg_cmd(video_path, geometry, crf, tmp_path),
            capture_output=True, text=True,
        )
        if (result.returncode != 0 or not os.path.isfile(tmp_path)) and _nvenc_available():
            # The GPU can be usable at probe time and busy/unavailable now.
            # One CPU retry is cheaper than losing the proxy entirely.
            _emit_cache_log(
                logger,
                f"NVENC encode failed for {os.path.basename(video_path)}; retrying on CPU",
                warning=True,
            )
            result = subprocess.run(
                _proxy_ffmpeg_cmd(video_path, geometry, crf, tmp_path, use_nvenc=False),
                capture_output=True, text=True,
            )
        if result.returncode != 0 or not os.path.isfile(tmp_path):
            _emit_cache_log(
                logger,
                f"Proxy encode failed for {os.path.basename(video_path)}: "
                f"{(result.stderr or '').strip()[-400:]}; decoding from source",
                warning=True,
            )
            return None

        # Build-time verification. This cannot be deferred to runtime: with
        # virtual chunking `total_frames` comes from `clip_frame_count`, so
        # nothing ever compares against the proxy's length except the silent
        # clamp in VideoDecode — a short proxy would feed duplicated trailing
        # frames forever without a single warning.
        proxy_frames = _proxy_frame_count(tmp_path)
        if proxy_frames != source_frames:
            _emit_cache_log(
                logger,
                f"Proxy for {os.path.basename(video_path)} has {proxy_frames} frames, "
                f"expected {source_frames}; discarding it and decoding from source",
                warning=True,
            )
            return None

        os.replace(tmp_path, proxy_path)
        _write_proxy_manifest(
            cache_dir, video_path, geometry, crf, source_frames, proxy_frames
        )
        return proxy_path
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def ensure_video_proxies(video_paths, geometry, crf=PROXY_CRF, workers=None, logger=None):
    """Build proxies for several videos in parallel.

    Returns ``{abs_source_path: abs_proxy_path}``, omitting videos whose proxy
    could not be built.
    """
    unique = []
    seen = set()
    for path in video_paths:
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            seen.add(abs_path)
            unique.append(abs_path)
    if not unique:
        return {}

    pending = [p for p in unique if not _valid_proxy(p, geometry, crf, len(_load_or_build_pts(p)))]
    proxies = {}
    for path in unique:
        if path not in pending:
            proxies[path] = _proxy_path(path, geometry, crf)

    if not pending:
        if proxies:
            _emit_cache_log(logger, f"Reusing {len(proxies)} decode proxy/proxies ({geometry.key})")
        return proxies

    # One job per source video now, not per window: these are long-running
    # full-length encodes, so fan out wider than the per-clip cache did.
    resolved = _resolve_proxy_workers(workers)
    parallel = min(max(resolved, 2), len(pending)) if shutil.which("ffmpeg") else 1
    _emit_cache_log(
        logger,
        f"Building {len(pending)} decode proxy/proxies ({geometry.key}) "
        f"with {parallel} worker(s)",
    )

    completed = 0
    report_every = _cache_progress_interval(len(pending))

    def run(path):
        try:
            return path, build_video_proxy(path, geometry, crf=crf, logger=logger)
        except Exception as exc:
            _emit_cache_log(
                logger,
                f"Proxy build raised for {os.path.basename(path)}: {exc}; decoding from source",
                warning=True,
            )
            return path, None

    def record(result):
        nonlocal completed
        path, proxy_path = result
        if proxy_path:
            proxies[path] = proxy_path
        completed += 1
        if completed == 1 or completed == len(pending) or completed % report_every == 0:
            _emit_cache_log(
                logger, f"Proxied {completed}/{len(pending)} video(s)"
            )

    if parallel == 1:
        for path in pending:
            record(run(path))
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(run, path) for path in pending]
            for future in as_completed(futures):
                record(future.result())
    return proxies


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
    essentially always VFR — see ``pts-based-frame-mapping.md (archived)``).
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


def enumerate_virtual_clips(video_path, clip_frames=768, clip_stem=None):
    """Enumerate fixed-length virtual windows over a video.

    Records the source metadata each window needs for timeline conversion
    (`source_video`, `source_frame_offset`, `source_start_seconds`,
    `source_pts_table`) without cutting any files. Frames are read at decode
    time from the source, or from its proxy when one exists.
    """
    video_path = os.path.abspath(video_path)
    video_name = clip_stem or Path(video_path).stem
    pts_array = _load_or_build_pts(video_path)
    total_frames = len(pts_array)
    if total_frames <= 0:
        return []

    cap = cv2.VideoCapture(video_path)
    avg_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if avg_fps is None or avg_fps <= 0:
        span = float(pts_array[-1] - pts_array[0]) if total_frames > 1 else 0.0
        avg_fps = (total_frames - 1) / span if span > 0 else 30.0

    pts_cache_path = _pts_cache_path(video_path)
    abs_pts_path = (
        os.path.abspath(pts_cache_path) if os.path.isfile(pts_cache_path) else None
    )

    clips = []
    num_clips = (total_frames + clip_frames - 1) // clip_frames
    for clip_idx in range(num_clips):
        start_frame = clip_idx * clip_frames
        actual_frames = min(clip_frames, total_frames - start_frame)
        if actual_frames <= 0:
            continue
        end_frame = start_frame + actual_frames - 1
        clip_info = {
            "clip_idx": clip_idx,
            "clip_name": f"{video_name}_clip_{clip_idx}",
            "frame": actual_frames,
            "duration": (
                float(pts_array[end_frame] - pts_array[start_frame]) + (1.0 / avg_fps)
            ),
            "source_video": video_path,
            "source_frame_offset": start_frame,
            "source_start_seconds": float(pts_array[start_frame] - pts_array[0]),
        }
        if abs_pts_path is not None:
            clip_info["source_pts_table"] = abs_pts_path
        clips.append(clip_info)
    return clips


def _process_video(
    video_path,
    csv_path,
    output_dir,
    proxy_geometry=None,
    proxy_crf=PROXY_CRF,
    proxy_workers=None,
    logger=None,
):
    """Process a single video: map CSV times to frames, build one dataset entry.

    No files are written and the video is not cut up. The entry covers the whole
    timeline and records `source_video` + `source_frame_offset`, so the loader
    reads frames straight out of the original.

    Windowing belongs to the dataset classes, not here. They slide over whatever
    timeline an entry describes, and `BehaviorTargetedSlidingDataset` needs the
    whole video to slide in: an entry pre-cut to `window_size` leaves no slack to
    shift a rare bout's phase into, so its targeted sampling degenerates to the
    plain sampler. Pre-cutting also drops every stretch that happens to contain no
    annotation, and truncates the bouts that straddle a cut.

    When `proxy_geometry` is given, a downscaled full-length proxy of the
    source is built once and recorded as `proxy_video`. Frames then decode
    from the small proxy while every timestamp still resolves against the
    original's PTS table. A failed or skipped proxy is not an error: the entry
    simply carries no `proxy_video` and decoding falls back to the source.

    Returns the entry dict, or None when the CSV holds no usable annotation.
    """
    video_name = Path(video_path).stem
    print(f"  Processing video: {video_name}")

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
    # for both CFR and VFR sources. See pts-based-frame-mapping.md (archived).
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

    video_annos = []
    for anno in annotations:
        first, last = anno["start_frame"], anno["end_frame"]
        if first > last:
            continue
        seg = [float(pts_array[first] - pts_array[0]),
               float(pts_array[last] - pts_array[0])]
        video_annos.append({
            "frame_segment": [first, last],
            "segment": seg,
            "timestamp_sec": list(seg),
            "label": anno["labelId"],
        })

    if not video_annos:
        print("    no usable annotations — skipped")
        return None

    # Duration = inter-frame span plus one frame's worth at the local average
    # fps, so a 30-frame video @30fps reports 1.0s and not 29/30s.
    entry = {
        "frame": total_frames,
        "duration": float(pts_array[-1] - pts_array[0]) + (1.0 / avg_fps),
        "annotations": video_annos,
        "source_video": os.path.abspath(video_path),
        "source_frame_offset": 0,
    }
    if os.path.isfile(pts_cache_path):
        # PTS table reference — present whenever the cache wrote successfully.
        # Loaders that don't know about it ignore it.
        entry["source_pts_table"] = os.path.abspath(pts_cache_path)
    print(f"    {len(video_annos)} bouts over {entry['duration']:.1f}s (kept whole)")

    if proxy_geometry is not None:
        proxy_path = build_video_proxy(
            video_path,
            proxy_geometry,
            crf=proxy_crf,
            source_frames=total_frames,
            logger=logger,
        )
        if proxy_path:
            # Decode shortcut. `source_*` above stays authoritative for every
            # timestamp; this only redirects which file the frames come from.
            entry["proxy_video"] = os.path.abspath(proxy_path)
            print(f"    Decoding from proxy ({proxy_geometry.key}): {proxy_path}")

    return entry


def _slice_entry(entry, pts_array, first, stop):
    """A copy of `entry` covering source frames [first, stop), times rebased.

    Returns None when the slice contains no annotation.
    """
    last = stop - 1
    if last < first:
        return None
    origin = float(pts_array[first])
    annotations = []
    for anno in entry["annotations"]:
        begin, finish = anno["frame_segment"]
        begin, finish = max(begin, first), min(finish, last)
        if begin > finish:
            continue
        seg = [float(pts_array[begin]) - origin, float(pts_array[finish]) - origin]
        annotations.append({
            "frame_segment": [begin - first, finish - first],
            "segment": seg,
            "timestamp_sec": list(seg),
            "label": anno["label"],
        })
    if not annotations:
        return None
    sliced = dict(entry)
    sliced["annotations"] = annotations
    sliced["frame"] = stop - first
    frame_seconds = entry["duration"] / max(entry["frame"], 1)
    sliced["duration"] = float(pts_array[last]) - origin + frame_seconds
    sliced["source_frame_offset"] = entry.get("source_frame_offset", 0) + first
    return sliced


def _split_one_video(name, entry, train_ratio):
    """Train/validation for a corpus of exactly one video: hold out its tail.

    Splitting whole videos is the rule, but with a single video that would leave
    one side empty. Cutting its timeline is the next best thing — the two halves
    are at least disjoint in time, which a window-level split of the same video
    would not be.
    """
    pts_array = _load_or_build_pts(entry["source_video"])
    cut = max(1, min(len(pts_array) - 1, int(len(pts_array) * train_ratio)))
    head = _slice_entry(entry, pts_array, 0, cut)
    tail = _slice_entry(entry, pts_array, cut, len(pts_array))
    if head is None or tail is None:
        # Every bout sits on one side of the cut; a validation half with no
        # annotation is worse than no split at all.
        return {name: {**entry, "subset": "train"}}
    return {
        f"{name}_head": {**head, "subset": "train"},
        f"{name}_tail": {**tail, "subset": "validation"},
    }


def _assign_subsets(entries, train_ratio):
    """Label every entry `train` or `validation`, splitting WHOLE videos.

    Never within a video. Adjacent windows of one recording show the same
    animals in the same cage seconds apart, so a window-level split puts near
    copies on both sides and reports a validation score the model has already
    effectively seen. Whole-video splitting is also what the published
    benchmarks do, which is what makes a number comparable.

    The shuffle is seeded, so the same corpus always splits the same way.
    """
    if len(entries) == 1:
        name, entry = next(iter(entries.items()))
        return _split_one_video(name, entry, train_ratio)

    order = sorted(entries)
    np.random.RandomState(42).shuffle(order)
    # Both sides get at least one video, whatever the ratio rounds to.
    n_train = min(len(order) - 1, max(1, round(len(order) * train_ratio)))
    return {
        name: {**entries[name], "subset": "train" if i < n_train else "validation"}
        for i, name in enumerate(order)
    }


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


def materialize_dataset_proxies(
    annotation_path,
    output_dir,
    geometry,
    crf=PROXY_CRF,
    workers=None,
    logger=None,
):
    """Give every source-backed entry in a dataset JSON a decode proxy.

    Builds one proxy per distinct ``source_video`` and rewrites the entries to
    point at it via ``proxy_video``. Entries without ``source_video`` are left
    alone: they already reference standalone clip files. Returns the path of
    the rewritten JSON, or ``annotation_path`` unchanged when there was nothing
    to do.
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    database = payload.get("database", {})

    sources = []
    for entry in database.values():
        source_video = entry.get("source_video")
        if source_video and os.path.isfile(source_video):
            sources.append(os.path.abspath(source_video))
    if not sources:
        return annotation_path

    proxies = ensure_video_proxies(
        sources, geometry, crf=crf, workers=workers, logger=logger
    )
    if not proxies:
        return annotation_path

    changed = False
    proxied_count = 0
    for entry in database.values():
        source_video = entry.get("source_video")
        if not source_video:
            continue
        proxy_path = proxies.get(os.path.abspath(source_video))
        if not proxy_path:
            continue
        proxied_count += 1
        if entry.get("proxy_video") != proxy_path:
            entry["proxy_video"] = proxy_path
            changed = True

    if not changed:
        return annotation_path

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    proxy_json = os.path.join(output_dir, "dataset_proxy.json")
    with open(proxy_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if logger is not None:
        logger.info(f"Proxied dataset entries: {proxied_count}/{len(database)}")
        logger.info(f"Proxied dataset JSON: {proxy_json}")
    return proxy_json


def prepare_dataset(dataset_path, train_ratio=0.8,
                    included_stems=None, explicit_pairs=None, output_dir=None,
                    proxy_geometry=DEFAULT_PROXY_GEOMETRY, proxy_crf=PROXY_CRF,
                    proxy_workers=None, logger=None):
    """Prepare a dataset directory for training.

    Scans for video+CSV pairs and generates dataset.json and classmap.txt.
    Skips if the output directory already exists with dataset.json.

    One entry per video, covering its whole timeline: entries are virtual,
    recording `source_video` + `source_frame_offset` rather than re-encoding
    anything. Windowing is the dataset classes' job at load time, not prep's.

    Args:
        dataset_path: Directory containing videos and CSV annotations.
        train_ratio: Fraction of *videos* for training (default: 0.8). The
            split is whole-video; see `_assign_subsets`.
        proxy_geometry: `ProxyGeometry` for the downscaled decode proxy built
            alongside each source video, or None to decode from the originals.
            Derive it from the training config with
            `vtrace.proxy_geometry.config_geometry` so the proxy matches
            what the pipeline will feed the model.
        proxy_crf: H.264 quality for the proxy encode.
        proxy_workers: Parallel proxy encodes (default: derived from CPU count).
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
        # Ensure classmap exists — regenerate from JSON if missing
        if not os.path.isfile(classmap_path):
            _generate_classmap_from_json(json_path, classmap_path)
            print(f"Generated classmap: {classmap_path}")
        print(f"Dataset already prepared at {output_dir}")
        return output_dir, json_path, classmap_path

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
    entries = {}
    for video_path, csv_path in pairs:
        entry = _process_video(
            video_path,
            csv_path,
            output_dir,
            proxy_geometry=proxy_geometry,
            proxy_crf=proxy_crf,
            proxy_workers=proxy_workers,
            logger=logger,
        )
        if entry is not None:
            entries[Path(video_path).stem] = entry
    if not entries:
        raise ValueError(f"None of the CSVs in {dataset_path} held a usable annotation.")

    database = _assign_subsets(entries, train_ratio)
    n_train = sum(1 for e in database.values() if e["subset"] == "train")
    print(f"Split: {n_train} train / {len(database) - n_train} validation "
          f"(whole videos)" if len(entries) > 1 else
          f"Split: 1 video, held out its last {1 - train_ratio:.0%} for validation")

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
