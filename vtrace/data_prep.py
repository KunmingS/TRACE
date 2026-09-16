"""Dataset auto-preparation: scan for videos+CSVs and generate annotations.

Given a dataset directory containing raw videos and per-video CSV annotations,
this module:
1. Writes dataset.json into a model artifact directory
2. Extracts class names from CSVs → classmap.txt
3. Records fixed-length training segments
4. Generates dataset.json in TRACE annotation format

Everything preparation leaves beside a source video goes into one folder,
``<video>.vtrace/`` (the full file name plus the suffix, so ``a.mp4`` and
``a.mkv`` in the same directory never share one): the per-frame PTS table
with its validation record, and the decode proxies under ``proxy/``. One
entry per video in a listing, whatever is cached for it.
"""

import csv
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import shutil
import subprocess
import threading
import time

import cv2
import numpy as np

from vtrace.console import (
    ACCENT, bar_markup, clear_status, console, duration, is_terminal, say,
    status,
)
from vtrace.model_artifacts import create_model_dir
from vtrace.proxy_geometry import ProxyGeometry


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# The folder beside each source video that holds everything cached for it.
SIDECAR_SUFFIX = ".vtrace"

# Decode proxies: one downscaled, frame-aligned copy of each source video.
# Bump PROXY_VERSION whenever the encode recipe changes, so every existing
# proxy is treated as stale without touching the manifest schema.
PROXY_VERSION = 1
PROXY_GOP = 30
PROXY_CRF = 23
DEFAULT_PROXY_GEOMETRY = ProxyGeometry(144, True)

# Probe results, keyed by the frame size probed: see `_nvenc_available`.
_NVENC_CACHE = {}


def _resolve_proxy_workers(workers=None):
    if workers is None:
        cpu = os.cpu_count() or 1
        return max(1, min(4, cpu // 4 if cpu >= 4 else 1))
    try:
        parsed = int(workers)
    except (TypeError, ValueError):
        return 1
    return max(1, min(8, parsed))


# ── Where prep's detail goes ─────────────────────────────────────────────────
# Preparing the CalMS21 demo indexes 89 videos, and the per-video detail —
# frame count, average rate, resolution, which proxy the frames will come from —
# is worth keeping and not worth watching go past. It is written to a file, and
# the terminal gets one line per video.
_detail_sink = None


def _detail(message: str) -> None:
    """Record `message` in the prep log, if one is open."""
    if _detail_sink is not None:
        _detail_sink.write(message + "\n")
        _detail_sink.flush()


class _prep_log:
    """Open `path` as the detail sink for the duration of the block."""

    def __init__(self, path):
        self.path = path
        self.handle = None

    def __enter__(self):
        global _detail_sink
        try:
            self.handle = open(self.path, "w", encoding="utf-8")
        except OSError:
            # A read-only output directory loses the detail, not the run.
            return self
        _detail_sink = self.handle
        return self

    def __exit__(self, *exc):
        global _detail_sink
        _detail_sink = None
        if self.handle is not None:
            self.handle.close()
        return False


def _emit_cache_log(logger, message, *, warning=False, detail=False):
    """Log a proxy-cache message.

    `detail=True` is for the cases that cost the reader nothing — reusing what
    is already built, counting what was enumerated. Building a proxy is slow
    enough to be worth announcing; finding one is not.
    """
    if logger is not None:
        if warning:
            logger.warning(message)
        elif detail:
            logger.debug(message)
        else:
            logger.info(message)
    elif not detail:
        level = "WARNING: " if warning else ""
        print(f"    {level}{message}")


# ── What a long encode says while it runs ────────────────────────────────────
# A full-length proxy of an overnight recording is hours of ffmpeg, and the
# frame count is known before it starts — so the wait can be a bar with an ETA
# rather than a blank screen. Three things belong on that line, because they
# are the three questions someone stares at a still terminal asking: what is
# running, how much longer, and did it get the GPU.
#
# On screen it is "making a smaller copy", not "encoding a proxy". Whoever is
# waiting on it is a biologist with a folder of recordings, and "proxy" is a
# word from the codebase, not from what they asked the tool to do. The line has
# to say what the wait buys in words that need no glossary.

# How often a non-terminal (a job queue's log, an SSE stream) gets a line. A
# redraw is free on a terminal and a new line is not, so the two differ by
# three orders of magnitude.
_ENCODE_LOG_INTERVAL = 120.0
_ENCODE_DRAW_INTERVAL = 0.25


def _encoder_label(use_nvenc):
    """How the encoder in use is named on the progress line."""
    return "GPU NVENC" if use_nvenc else "CPU x264"


class _EncodeProgress:
    """The line one ffmpeg proxy encode redraws while it runs.

    On a terminal this is a single line redrawn in place and wiped at the end —
    the encode's own progress stops being true the moment it finishes, and the
    line worth keeping is the summary the caller prints. Everywhere else (a job
    queue capturing stdout, the annotator's SSE log) there is nothing to redraw
    into, so the same information goes out as an ordinary line every couple of
    minutes instead.
    """

    def __init__(self, total_frames, encoder, prefix="", logger=None):
        self.total = max(1, int(total_frames))
        self.encoder = encoder
        self.prefix = prefix
        self.logger = logger
        self.live = is_terminal()
        self.started = time.monotonic()
        self.last_draw = 0.0
        self.last_log = self.started
        self.frames = 0

    def _eta(self, now):
        """Seconds left, from our own elapsed/frames rather than ffmpeg's
        `speed`: an encode that spends its first minutes on a slow stretch of
        the file would otherwise show an ETA that walks backwards."""
        if self.frames <= 0:
            return None
        elapsed = now - self.started
        return elapsed / self.frames * (self.total - self.frames)

    def update(self, frames, speed=None):
        self.frames = min(int(frames), self.total)
        now = time.monotonic()
        eta = self._eta(now)
        percent = int(self.frames * 100 / self.total)
        tail = self.encoder if not speed or speed == "N/A" else f"{self.encoder} {speed}"

        if self.live:
            if now - self.last_draw < _ENCODE_DRAW_INTERVAL:
                return
            self.last_draw = now
            eta_text = f"eta {duration(eta)}" if eta is not None else "eta --"
            status(f"  [dim]{self.prefix}making a smaller copy[/]  "
                   f"{bar_markup(self.frames / self.total)}  "
                   f"[{ACCENT}]{percent:>3d}%[/]  [dim]{eta_text}[/]  [dim]{tail}[/]")
            return

        if now - self.last_log < _ENCODE_LOG_INTERVAL:
            return
        self.last_log = now
        eta_text = f", ~{duration(eta)} left" if eta is not None else ""
        _emit_cache_log(
            self.logger,
            f"Making a smaller copy: {percent}% of {self.total:,} frames"
            f"{eta_text} ({tail})",
        )

    def close(self):
        if self.live:
            clear_status()

    @property
    def elapsed(self):
        return time.monotonic() - self.started


def _run_ffmpeg_with_progress(cmd, progress):
    """Run `cmd`, feeding `progress` from ffmpeg's own `-progress` stream.

    Returns ``(returncode, stderr_text)``, the two things the caller used to
    take from `subprocess.run`. stdout carries the progress blocks and so can no
    longer be captured wholesale; stderr is drained by a thread because a full
    pipe on either stream would deadlock the other.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    errors = []

    def drain_stderr():
        try:
            for line in proc.stderr:
                errors.append(line)
        except (OSError, ValueError):
            pass

    reader = threading.Thread(target=drain_stderr, daemon=True)
    reader.start()
    speed = None
    try:
        for line in proc.stdout:
            key, sep, value = line.strip().partition("=")
            if not sep:
                continue
            if key == "speed":
                speed = value.strip()
            elif key == "frame" and progress is not None:
                try:
                    progress.update(int(value), speed)
                except ValueError:
                    pass
    except BaseException:
        # A Ctrl-C (or the SIGHUP a dropped terminal delivers) must not leave a
        # multi-hour encode running unattended and unreferenced.
        proc.kill()
        raise
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass
        proc.wait()
        reader.join(timeout=2)
        try:
            proc.stderr.close()
        except OSError:
            pass
    return proc.returncode, "".join(errors)


def _cache_progress_interval(total):
    return max(1, min(25, total // 20 if total >= 20 else 1))


def _source_signature(video_path):
    stat = os.stat(video_path)
    return {
        "source_path": os.path.abspath(video_path),
        "source_mtime_ns": stat.st_mtime_ns,
        "source_size": stat.st_size,
    }


def sidecar_dir(video_path):
    """``<video>.vtrace``: the one folder holding everything cached for a video.

    Built from the path as given (no ``abspath``), so a relative source keeps a
    relative sidecar and the callers that record absolute paths resolve them
    themselves, as they did when the files sat directly beside the video.
    """
    return os.fspath(video_path) + SIDECAR_SUFFIX


def _proxy_cache_dir(video_path, geometry, crf):
    """Directory holding one decode proxy of ``video_path``.

    ``<video>.vtrace/proxy/<geometry>_crf<crf>_g<gop>_v<version>``. The key
    deliberately omits the clip/window length: a proxy is full-length, so
    changing ``window_size`` no longer invalidates it (the old per-window clip
    cache keyed on ``f{frames}`` and had to be rebuilt from scratch).
    ``_v{PROXY_VERSION}`` lets a change to the encode recipe invalidate every
    proxy without migrating the manifest schema.
    """
    video_path = os.path.abspath(video_path)
    cache_key = f"{geometry.key}_crf{int(crf)}_g{PROXY_GOP}_v{PROXY_VERSION}"
    return os.path.join(sidecar_dir(video_path), "proxy", cache_key)


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


def _nvenc_available(size=None):
    """Whether NVENC can encode a frame of `size` square pixels here.

    Listing the encoder is not enough: ffmpeg advertises `h264_nvenc` from the
    build config, so a box with a driver/library mismatch or no GPU still
    reports it and then fails at encode time with "No capable devices found".
    Probe by encoding one synthetic frame.

    The size is part of the question, not a detail of how it is asked. NVENC
    refuses frames below a minimum dimension — around 160px on current cards —
    with "Frame Dimension less than the minimum supported value", so a probe at
    some token size answers "is there a GPU" while the caller asked "can this
    encode run on it". A 64x64 probe fails on every card ever made, which is
    how a box with two working GPUs came to encode all of its proxies on the
    CPU while reporting nothing amiss.

    Proxies are square or wider, so probing `size` x `size` is the conservative
    case: a geometry whose square passes will pass at its real, wider size.
    Cached per size, since a run mixes at most a couple of them.
    """
    key = int(size) if size else 256
    if key not in _NVENC_CACHE:
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", f"color=black:s={key}x{key}:d=1",
                 "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, text=True, timeout=30,
            )
            _NVENC_CACHE[key] = result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _NVENC_CACHE[key] = False
    return _NVENC_CACHE[key]


def _proxy_ffmpeg_cmd(video_path, geometry, crf, out_path, use_nvenc=None):
    """ffmpeg args for a frame-aligned decode proxy.

    Three settings are load-bearing:

    * ``-fps_mode passthrough`` — the default (``cfr``) duplicates or drops
      frames on VFR input, which would destroy the 1:1 frame alignment the
      whole proxy design rests on. TRACE targets VFR webcam recordings.
    * ``setpts=N/(30*TB)`` — renumbers output PTS monotonically by frame index.
      VFR sources with duplicate timestamps otherwise make the mp4 muxer emit
      non-monotonic DTS. Safe because the proxy's own PTS are never read: the
      source's ``pts.npy`` stays the timeline authority.
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
    if _nvenc_available(geometry.short_side) if use_nvenc is None else use_nvenc:
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
        # Frame counts on stdout, as key=value blocks, so the wait can be a bar.
        # `-nostats` drops the human-readable line ffmpeg would also write.
        "-nostats", "-progress", "pipe:1",
        os.fspath(out_path),
    ]
    return cmd


def build_video_proxy(video_path, geometry, crf=PROXY_CRF, source_frames=None,
                      logger=None, prefix="", show_progress=True):
    """Build (or reuse) the decode proxy for one video.

    Returns the absolute proxy path, or None when no usable proxy exists — a
    missing proxy is never fatal, callers simply decode from the source.

    `show_progress` draws the encode's bar on the one status line. Callers that
    run several encodes at once pass False: two bars sharing one line render as
    neither. `prefix` is what goes in front of the label, so a bar that appears
    part-way through a dataset still says which video it belongs to.
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
    # Resolved here rather than left to `_proxy_ffmpeg_cmd`, so the progress
    # line can name the encoder that is actually about to run.
    use_nvenc = _nvenc_available(geometry.short_side)

    def encode(nvenc):
        progress = None
        if show_progress:
            progress = _EncodeProgress(
                source_frames, _encoder_label(nvenc), prefix=prefix, logger=logger
            )
            # Drawn before ffmpeg has reported a single frame: the answer to
            # "is it using the GPU" should not wait on the first progress block.
            progress.update(0)
        try:
            return _run_ffmpeg_with_progress(
                _proxy_ffmpeg_cmd(video_path, geometry, crf, tmp_path, use_nvenc=nvenc),
                progress,
            ), progress
        finally:
            if progress is not None:
                progress.close()

    try:
        (returncode, stderr), progress = encode(use_nvenc)
        if (returncode != 0 or not os.path.isfile(tmp_path)) and use_nvenc:
            # The GPU can be usable at probe time and busy/unavailable now.
            # One CPU retry is cheaper than losing the proxy entirely.
            _emit_cache_log(
                logger,
                f"NVENC encode failed for {os.path.basename(video_path)}; retrying on CPU",
                warning=True,
            )
            use_nvenc = False
            (returncode, stderr), progress = encode(False)
        if returncode != 0 or not os.path.isfile(tmp_path):
            _emit_cache_log(
                logger,
                f"Proxy encode failed for {os.path.basename(video_path)}: "
                f"{(stderr or '').strip()[-400:]}; decoding from source",
                warning=True,
            )
            return None

        # Build-time verification. This cannot be deferred to runtime: with
        # virtual chunking `total_frames` comes from `clip_frame_count`, so
        # nothing ever compares against the proxy's length except the silent
        # clamp in VideoDecode — a short proxy would feed duplicated trailing
        # frames forever without a single warning.
        if show_progress:
            # Indexing a few million frames is not instant either, and a line
            # that still said "encoding" through it would be a lie.
            status(f"  [dim]{prefix}checking the copy[/]")
        proxy_frames = _proxy_frame_count(tmp_path)
        if show_progress:
            clear_status()
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
        # The bar is wiped when it ends, so an encode long enough to have been
        # waited on leaves one line behind saying what the wait bought. A short
        # one leaves nothing: 89 demo clips should not print 89 lines.
        if progress is not None and progress.elapsed >= 60:
            _emit_cache_log(
                logger,
                f"Made a smaller copy of {os.path.basename(video_path)} in "
                f"{duration(progress.elapsed)} ({_encoder_label(use_nvenc)}, "
                f"{source_frames:,} frames)",
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
            _emit_cache_log(logger, f"Reusing {len(proxies)} decode proxy/proxies "
                            f"({geometry.key})", detail=True)
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
            return path, build_video_proxy(
                path, geometry, crf=crf, logger=logger,
                show_progress=(parallel == 1),
            )
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


def pts_table_path(video_path):
    """``<video>.vtrace/pts.npy``: the cached per-frame PTS table of a source video."""
    return os.path.join(sidecar_dir(video_path), "pts.npy")


# The name the rest of the package grew up with.
_pts_cache_path = pts_table_path


def _pts_meta_path(video_path):
    """``<video>.vtrace/pts.meta.json``: the source's mtime+size at the moment
    the PTS table was built. Lets us validate the table from `os.stat` alone,
    without re-opening the container."""
    return os.path.join(sidecar_dir(video_path), "pts.meta.json")


def _load_or_build_pts(video_path):
    """Return per-frame PTS array (seconds, ``float64``) for ``video_path``.

    The array carries one entry per encoded frame: ``pts[i]`` is the
    presentation timestamp of frame ``i`` as recorded in the container. This
    is the canonical source of truth for time ↔ frame conversion and works
    correctly for both CFR and VFR sources (USB webcams in dim labs are
    essentially always VFR — see ``pts-based-frame-mapping.md (archived)``).
    Building it via decord reads only the index (no decoding), so it costs
    seconds even for hours of footage.

    The result is cached as ``pts.npy`` in the source's ``.vtrace`` folder,
    paired with a ``pts.meta.json`` record of the source's mtime (ns) and
    size at build time.

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
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
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
                    f"Invalid pair spec '{spec}'. Use VIDEO_PATH=ANNOTATION_PATH "
                    f"(a .csv, or a .json from an earlier annotator)."
                )
            video_spec, csv_spec = (part.strip() for part in spec.split("=", 1))
        if not video_spec or not csv_spec:
            raise ValueError(
                f"Invalid pair spec '{spec_label}'. Use VIDEO_PATH=ANNOTATION_PATH "
                f"(a .csv, or a .json from an earlier annotator)."
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
            if not csv_path.lower().endswith(ANNOTATION_EXTENSIONS):
                raise ValueError(
                    f"Pair annotation must be a .csv or a .json file: {csv_path}")
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"Pair annotation file not found: {csv_path}")
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
    # `<stem>.json` from an earlier annotator, taken only when the stem has no
    # CSV at all: a CSV beside it is the migrated copy and is the one to trust.
    jsons_by_stem = {}
    for index, fname in enumerate(entries):
        lower = fname.lower()
        if lower.endswith(".json"):
            jsons_by_stem.setdefault(fname[:-5], fname)
            continue
        if not lower.endswith(".csv"):
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
            if stem in jsons_by_stem:
                pairs.append((video_path, os.path.join(dataset_path, jsons_by_stem[stem])))
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

    Column names are stripped. A CSV padded so its columns line up —
    `labelId,  timestamp,  endTimestamp` — is a reasonable thing to write or
    edit by hand, and the values already survive it (`float` skips leading
    whitespace, and every label goes through `.strip()`). Without this the
    header alone would turn such a file into a `KeyError` on 'timestamp', which
    says nothing about the two spaces that caused it. The annotator's own
    reader has always trimmed here; this is the Python side catching up.
    """
    reader = csv.DictReader(
        line for line in file_obj if not line.lstrip().startswith("#"))
    if reader.fieldnames:
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
    return reader


ANNOTATION_EXTENSIONS = (".csv", ".json")


def _json_annotation_entry(payload, path, video_stem):
    """The per-video entry inside an annotator JSON file.

    Two shapes are accepted: `{"database": {<video key>: {...}}}` — the
    annotator's own file, whether it holds one video (`<video>.json`) or many
    (`annotations.json`) — and a bare entry `{"annotations": [...]}`. In a
    database the entry is looked up by the video's stem; a file holding exactly
    one video is taken as that video's regardless of key, since a renamed video
    beside its own JSON is the common case, not a mismatch.
    """
    if isinstance(payload, dict) and isinstance(payload.get("database"), dict):
        database = payload["database"]
        if video_stem in database:
            return database[video_stem]
        if len(database) == 1:
            return next(iter(database.values()))
        keys = ", ".join(sorted(database)[:5])
        raise ValueError(
            f"{path} holds annotations for {len(database)} videos and none is "
            f"keyed '{video_stem}' (keys start: {keys}).")
    if isinstance(payload, dict) and isinstance(payload.get("annotations"), list):
        return payload
    raise ValueError(f"{path} is not an annotation file the annotator writes.")


def _read_annotation_rows(path, video_stem=None):
    """Rows of `labelId / timestamp / endTimestamp / review` from a CSV or a JSON.

    The CSV is the annotator's current format. The JSON is what earlier
    versions wrote — `frame_segment`, `time_segment`, `label` per bout — and a
    folder labelled before the switch should train without a conversion step.
    Seconds come from `time_segment`; a bout that only has frames is placed
    with the entry's fps, the way the annotator itself does when it loads one.
    """
    if not str(path).lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return [dict(row) for row in _csv_dict_reader(f)]
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    entry = _json_annotation_entry(payload, path, video_stem)
    fps = float(entry.get("fps") or 0.0)
    rows = []
    for anno in entry.get("annotations") or []:
        if not isinstance(anno, dict):
            continue
        times = anno.get("time_segment")
        frames = anno.get("frame_segment")
        if isinstance(times, (list, tuple)) and len(times) == 2:
            start, end = float(times[0]), float(times[1])
        elif isinstance(frames, (list, tuple)) and len(frames) == 2 and fps > 0:
            start, end = float(frames[0]) / fps, (float(frames[1]) + 1.0) / fps
        else:
            continue
        rows.append({
            "labelId": str(anno.get("label") or ""),
            "timestamp": start,
            "endTimestamp": end,
            "review": str(anno.get("review") or ""),
        })
    return rows


def _is_rejected(row):
    """True for a row the reviewer threw out in the annotator.

    A rejected prediction is a false positive. It stays in the CSV so the decision
    travels with the dataset, but training must never see it. Rows without the
    column are never rejected.
    """
    return (row.get("review") or "").strip().lower() == "rejected"


def _extract_classes(pairs):
    """Every label across the (video, annotation file) pairs, sorted."""
    labels = set()
    for video_path, ann_path in pairs:
        labels |= _annotation_labels(video_path, ann_path)
    return sorted(labels)


# ── splitting a corpus ─────────────────────────────────────────────────────
# Videos labelled with a rare behavior are rare themselves, and a random cut of
# a small corpus routinely lands every one of them on the same side — a
# validation set with no `attack` in it cannot tell you anything about `attack`.
# So the cut is stratified on which behaviors each video contains.

# Videos with no behavior at all still have to be shared out proportionally, so
# they are stratified under a label of their own.
_BG_STRATUM = "(background)"


def _annotation_labels(video_path, ann_path):
    """The set of behaviors annotated for one video (empty => background only)."""
    labels = set()
    for row in _read_annotation_rows(ann_path, Path(video_path).stem):
        if _is_rejected(row):
            continue
        label = (row.get("labelId") or "").strip()
        if label:
            labels.add(label)
    return labels


def stratified_split(pairs, ratios, seed=42):
    """Divide video/CSV pairs into named splits with even per-behavior shares.

    The multi-label iterative stratification of Sechidis, Tsoumakas & Vlahavas
    (2011): repeatedly take the behavior with the fewest videos still
    unassigned, and give each of those videos to whichever split is furthest
    from its quota for that behavior. Starting from the rarest keeps the classes
    that a random cut would strand — there is no slack left by the time the
    common ones are placed, and there does not need to be.

    Args:
        pairs: ``(video, csv)`` tuples or ``"VIDEO=CSV"`` specs — whichever form
            comes in is the form that comes back.
        ratios: ``[(split_name, fraction)]``; fractions are normalized.
        seed: makes tie-breaking deterministic, so a dataset always splits the
            same way and a run stays reproducible.

    Returns:
        ``{split_name: [pair]}``, each list in input order.
    """
    import random
    from collections import defaultdict

    total_fraction = sum(fraction for _, fraction in ratios)
    if total_fraction <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    names = [name for name, _ in ratios]
    share = {name: fraction / total_fraction for name, fraction in ratios}

    def parts(pair):
        if isinstance(pair, (tuple, list)):
            return str(pair[0]), str(pair[1])
        video, _, csv_path = str(pair).partition("=")
        return video.strip(), csv_path.strip()

    strata = {}
    pool = defaultdict(set)
    for pair in pairs:
        video_path, csv_path = parts(pair)
        labels = _annotation_labels(video_path, csv_path) or {_BG_STRATUM}
        strata[video_path] = labels
        for label in labels:
            pool[label].add(video_path)

    rng = random.Random(seed)
    quota = {name: share[name] * len(pairs) for name in names}
    label_quota = {label: {name: share[name] * len(keys) for name in names}
                   for label, keys in pool.items()}
    assignment = {}
    unassigned = set(strata)

    def take(video_path, name):
        assignment[video_path] = name
        unassigned.discard(video_path)
        quota[name] -= 1
        for label in strata[video_path]:
            if video_path in pool[label]:
                pool[label].discard(video_path)
                label_quota[label][name] -= 1

    while unassigned:
        candidates = [label for label, keys in pool.items() if keys]
        if not candidates:
            break
        rarest = min(candidates, key=lambda label: (len(pool[label]), label))
        for video_path in sorted(pool[rarest]):
            if video_path not in unassigned:
                continue
            take(video_path, max(names, key=lambda name: (
                label_quota[rarest][name], quota[name], rng.random())))

    # Anything whose every stratum emptied first: place it on overall quota.
    for video_path in sorted(unassigned):
        take(video_path, max(names, key=lambda name: (quota[name], rng.random())))

    result = {name: [] for name in names}
    for pair in pairs:
        result[assignment[parts(pair)[0]]].append(pair)
    return result


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
    prefix="",
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
    _detail(f"Processing video: {video_name}")

    # Load the annotations — the CSV the annotator writes now, or the JSON an
    # earlier version wrote; `_read_annotation_rows` makes them the same rows.
    annotations = []
    rejected = 0
    for row in _read_annotation_rows(csv_path, video_name):
        if _is_rejected(row):
            rejected += 1
            continue
        annotations.append({
            "labelId": str(row["labelId"]).strip(),
            "timestamp": float(row["timestamp"]),
            "endTimestamp": float(row["endTimestamp"]),
        })
    if rejected:
        print(f"    {len(annotations)} annotations ({rejected} rejected, skipped)")
    else:
        _detail(f"  {len(annotations)} annotations")

    # Build / load the PTS table — one canonical timestamp per encoded frame.
    # Replaces the previous cv2 CAP_PROP_POS_MSEC per-frame loop and is correct
    # for both CFR and VFR sources. See pts-based-frame-mapping.md (archived).
    _detail("  Loading PTS table ...")
    # Minutes of index scanning on a multi-hour source, with nothing to count:
    # decord reports no progress until it is done. So the line says which step
    # is running and leaves it at that.
    status(f"  [dim]{prefix}reading timestamps[/]  [dim]{video_name}[/]")
    pts_array = _load_or_build_pts(video_path)
    clear_status()
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
    _detail(f"  {total_frames} frames, avg_fps={avg_fps:.3f}, {width}x{height}")

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
        _detail("  no usable annotations — skipped")
        print(f"  {video_name}: no usable annotations — skipped")
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
    _detail(f"  {len(video_annos)} bouts over {entry['duration']:.1f}s (kept whole)")

    if proxy_geometry is not None:
        proxy_path = build_video_proxy(
            video_path,
            proxy_geometry,
            crf=proxy_crf,
            source_frames=total_frames,
            logger=logger,
            prefix=prefix,
        )
        if proxy_path:
            # Decode shortcut. `source_*` above stays authoritative for every
            # timestamp; this only redirects which file the frames come from.
            entry["proxy_video"] = os.path.abspath(proxy_path)
            _detail(f"  Decoding from proxy ({proxy_geometry.key}): {proxy_path}")

    return entry


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


TRAIN_SUBSET = "train"
VALIDATION_SUBSET = "validation"


def _print_corpus_summary(entries, class_map):
    """What the corpus turned out to contain, one row per behavior.

    Fifty-five lines of "this video had 31 bouts" is a file's worth of detail —
    it goes to prep.log. What is worth looking at before a three-hour run is
    whether every behavior actually has enough of itself to learn from, and
    whether the split left any of them thin. That is a table.
    """
    videos = Counter()
    bouts = Counter()
    labelled = Counter()
    for entry in entries.values():
        seen = set()
        for anno in entry["annotations"]:
            label = anno["label"]
            bouts[label] += 1
            start, end = anno["segment"]
            labelled[label] += max(0.0, end - start)
            seen.add(label)
        for label in seen:
            videos[label] += 1

    total_video = sum(entry["duration"] for entry in entries.values())
    active = console()
    if active is None:
        print(f"  {'behavior':<20}{'videos':>7}{'bouts':>8}{'labelled':>12}")
        for label in class_map:
            print(f"  {label:<20}{videos[label]:>7}{bouts[label]:>8}"
                  f"{duration(labelled[label]):>12}")
        print(f"  {'total':<20}{len(entries):>7}{sum(bouts.values()):>8}"
              f"{duration(sum(labelled.values())):>12}"
              f"   in {duration(total_video)} of video")
        return

    from rich.table import Table

    table = Table.grid(padding=(0, 2))
    table.add_column(width=20)
    table.add_column(justify="right", width=6)
    table.add_column(justify="right", width=6)
    table.add_column(justify="right", width=10)
    table.add_column()
    table.add_row("  [dim]behavior[/]", "[dim]videos[/]", "[dim]bouts[/]",
                  "[dim]labelled[/]", "")
    for label in class_map:
        table.add_row(f"  {label}", f"[{ACCENT}]{videos[label]}[/]",
                      f"[{ACCENT}]{bouts[label]}[/]",
                      duration(labelled[label]), "")
    table.add_row("  [bold]total[/]", f"[bold]{len(entries)}[/]",
                  f"[bold]{sum(bouts.values())}[/]",
                  f"[bold]{duration(sum(labelled.values()))}[/]",
                  f"[dim]in {duration(total_video)} of video[/]")
    active.print(table)


def prepare_dataset(dataset_path, subset=TRAIN_SUBSET,
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
        subset: Which subset every entry belongs to — `train` or `validation`.
            Nothing is split here: a corpus is training data or it is evaluation
            data, and which one it is was decided by the picker that chose the
            folder. Holding a slice of the training videos back would only be
            the same decision made worse, since it is made without seeing them.
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
                f"Expected matching video ({exts}) and annotation files "
                f"(.csv, or .json from an earlier annotator) for each stem."
            )
        raise FileNotFoundError(
            f"No video+annotation pairs found in {dataset_path}. "
            f"Expected matching video ({exts}) and annotation files "
            "(e.g., video1.mp4 + video1.csv, or video1.mkv + video1.json)."
        )
    say(f"[bold]Preparing the {subset} dataset[/]  "
        f"[{ACCENT}]{len(pairs)}[/][dim] video"
        f"{'s' if len(pairs) != 1 else ''}[/]")

    # Extract class map from every annotation file, CSV or JSON
    class_map = _extract_classes(pairs)
    if not class_map:
        raise ValueError(f"No labels found in the annotation files in {dataset_path}")

    # Process each video. The detail goes to prep.log beside the dataset; the
    # terminal gets a line per video, so 89 of them is 89 lines and not 500.
    os.makedirs(output_dir, exist_ok=True)
    detail_path = Path(output_dir) / "prep.log"
    entries = {}
    counter_width = len(str(len(pairs)))
    with _prep_log(detail_path):
        for index, (video_path, csv_path) in enumerate(pairs, 1):
            counter = f"{index:>{counter_width}d}/{len(pairs)}"
            entry = _process_video(
                video_path,
                csv_path,
                output_dir,
                proxy_geometry=proxy_geometry,
                proxy_crf=proxy_crf,
                proxy_workers=proxy_workers,
                logger=logger,
                prefix=f"{counter}  ",
            )
            if entry is None:
                continue
            entries[Path(video_path).stem] = entry
            line = (f"  {Path(video_path).stem}   "
                    f"{len(entry['annotations'])} bouts, "
                    f"{entry['duration']:.0f}s")
            _detail(line)
            # Transient: prep takes a minute and should say where it is, but
            # a line per video is the wall this replaced.
            status(f"  [dim]indexing[/] [{ACCENT}]{counter}[/]  "
                   f"[dim]{Path(video_path).stem}[/]")
    clear_status()
    if not entries:
        raise ValueError(f"None of the CSVs in {dataset_path} held a usable annotation.")

    _print_corpus_summary(entries, class_map)

    database = {name: {**entry, "subset": subset} for name, entry in entries.items()}

    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"database": database}, f, indent=2, ensure_ascii=False)
    _detail(f"Dataset JSON: {json_path}")

    # Write classmap
    with open(classmap_path, "w", encoding="utf-8") as f:
        for name in class_map:
            f.write(name + "\n")
    _detail(f"Class map: {classmap_path}")

    # Which files the index is made of is in the log; what the reader needs
    # here is that it worked, and where to look if it did not.
    say(f"[bold]Prepared[/] [{ACCENT}]{len(database)}[/] "
        f"video{'s' if len(database) != 1 else ''}"
        f"[dim]  — detail in {detail_path}[/]")

    return output_dir, json_path, classmap_path
