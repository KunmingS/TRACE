"""TRACE annotation server — FastAPI application.

Handles video file serving, directory browsing, codec detection,
HEVC→H.264 transcoding, label management, and serves the bundled
frontend in production mode.
"""
import asyncio
import logging
import os
import re
import subprocess
import time
import json
from typing import Optional, List, Dict, Tuple
import csv

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
import mimetypes

logger = logging.getLogger(__name__)

app = FastAPI(title="TRACE", description="Temporal action detection for animal behavior")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHUNK_SIZE = int(os.environ.get('VIDEO_CHUNK_SIZE', 2 * 1024 * 1024))  # 2MB default
UPLOAD_CHUNK_SIZE = int(os.environ.get('UPLOAD_CHUNK_SIZE', 4 * 1024 * 1024))  # 4MB default
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
ALLOWED_LABEL_EXTENSIONS = {'.csv'}

# Codecs that need a full re-encode for browser playback (AV1/VP9 omitted — all
# target browsers play them natively). The frontend decides what to do per-file
# via the `action` query param; these constants are only the server-side safety net.
_NEEDS_TRANSCODE_CODECS = {
    'hevc', 'h265',
    'mpeg4', 'mpeg2video',
    'msmpeg4v3', 'wmv3', 'flv1',
}

# Safety allowlist for action='raw': codec ∈ this set AND container ∈ _RAW_CAPABLE_CONTAINERS.
# A confused client can't trick the server into serving something no browser can demux.
_RAW_CAPABLE_CODECS = {'h264', 'avc1', 'hevc', 'h265', 'vp9', 'av1'}
_RAW_CAPABLE_CONTAINERS = {'mp4', 'mov', 'webm'}

# {file_path: (mtime, info_or_None)} — keyed by absolute path.
_video_info_cache: Dict[str, Tuple[float, Optional[Dict[str, str]]]] = {}

_nvenc_cache: Optional[bool] = None


def _normalize_container(raw_format_name: str, file_ext: str = '') -> str:
    """Pick a canonical container name from ffprobe's comma-separated list.

    ffprobe reports both `.mkv` and `.webm` files as `matroska,webm` because
    WebM is structurally a Matroska subset. Use the file extension as the
    primary disambiguator, falling back to a priority order.
    """
    parts = [p.strip() for p in raw_format_name.split(',') if p.strip()]
    ext = file_ext.lower().lstrip('.')
    if ext == 'mkv' and 'matroska' in parts:
        return 'matroska'
    if ext in parts:
        return ext
    for preferred in ('mp4', 'webm', 'matroska', 'mov', 'avi'):
        if preferred in parts:
            return preferred
    return parts[0] if parts else ''


def _parse_rational(value: str) -> Optional[float]:
    """Parse an ffprobe rational like ``30/1`` or ``1500000/50001`` to float.

    Returns ``None`` for missing / zero / malformed values, so callers can
    short-circuit instead of inheriting a divide-by-zero or NaN.
    """
    if not value:
        return None
    parts = value.split('/')
    try:
        if len(parts) == 2:
            num = float(parts[0])
            den = float(parts[1])
            if den == 0:
                return None
            return num / den
        return float(value)
    except (ValueError, TypeError):
        return None


def _get_video_info(file_path: str) -> Optional[Dict[str, object]]:
    """Detect video codec, container, and frame-rate metadata via ffprobe.

    Cached on ``(path, mtime)``.

    The returned dict carries:

    - ``codec`` (str) — e.g. ``'h264'``, ``'hevc'``.
    - ``container`` (str) — disambiguated via ``_normalize_container``.
    - ``rFrameRate`` (float | None) — ffprobe's ``r_frame_rate``: the
      "real base frame rate" reported in the container (typically the
      nominal capture fps).
    - ``avgFrameRate`` (float | None) — ffprobe's ``avg_frame_rate``:
      ``num_frames / duration``, computed across the actual presented
      timestamps.
    - ``isVfr`` (bool | None) — true when the two rates disagree by more
      than ~1% (= classic VFR signature, especially USB webcams in dim
      labs whose auto-exposure drops frames). Frontend uses this for a
      small VFR badge in FileBrowser; backend uses it as advisory only —
      time ↔ frame math goes through the per-frame PTS table built by
      ``data_prep`` (see ``docs/pts-based-frame-mapping.md``).
    """
    try:
        mtime = os.stat(file_path).st_mtime
    except OSError:
        return None

    cached = _video_info_cache.get(file_path)
    if cached and cached[0] == mtime:
        return cached[1]

    info: Optional[Dict[str, object]] = None
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
             '-show_entries',
             'stream=codec_name,r_frame_rate,avg_frame_rate:format=format_name',
             '-of', 'json', file_path],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
        fmt = data.get('format', {})
        if streams:
            stream0 = streams[0]
            codec = stream0.get('codec_name', '').lower() or None
            raw_fmt = fmt.get('format_name', '').lower()
            ext = os.path.splitext(file_path)[1]
            container = _normalize_container(raw_fmt, ext) if raw_fmt else ''
            r_fr = _parse_rational(stream0.get('r_frame_rate', ''))
            avg_fr = _parse_rational(stream0.get('avg_frame_rate', ''))
            # Classic VFR signature: r_frame_rate (nominal) and
            # avg_frame_rate (= n_frames / duration) disagree. ~1% slack
            # because rounding in the duration field produces tiny
            # disagreements even on clean CFR sources.
            is_vfr: Optional[bool] = None
            if r_fr and avg_fr and r_fr > 0 and avg_fr > 0:
                is_vfr = abs(r_fr - avg_fr) / max(r_fr, avg_fr) > 0.01
            if codec:
                info = {
                    'codec': codec,
                    'container': container,
                    'rFrameRate': r_fr,
                    'avgFrameRate': avg_fr,
                    'isVfr': is_vfr,
                }
    except Exception as e:
        logger.warning(f"ffprobe failed for {file_path}: {e}")

    _video_info_cache[file_path] = (mtime, info)
    return info


def _get_video_codec(file_path: str) -> Optional[str]:
    """Backward-compat shim — returns just the codec name."""
    info = _get_video_info(file_path)
    return info['codec'] if info else None


def _get_transcoded_path(file_path: str) -> str:
    """Path for the cached H.264 transcoded version.

    Includes the original extension in the cache name so `foo.mp4` and `foo.mkv`
    in the same directory don't collide (they'd otherwise share a cache).
    """
    return file_path + '.h264.mp4'


def _get_remuxed_path(file_path: str) -> str:
    """Path for the cached MP4-remuxed version (codec stream-copied).

    Same anti-collision rationale as `_get_transcoded_path`.
    """
    return file_path + '.remux.mp4'


# Output as fragmented MP4 with a global sidx, instead of `+faststart`
# non-fragmented MP4. Two reasons:
#
# 1. `+empty_moov+frag_keyframe+default_base_moof` keeps the moov atom a
#    few KB (the per-sample tables would otherwise grow linearly with frame
#    count and reach tens of MB on hours-long footage; the browser must
#    download all of moov before `canplay`).
# 2. `+global_sidx` writes a single segment index near the start of the
#    file mapping presentation time → fragment byte offset. Without it the
#    browser has no time→byte map and walks every moof box from the head
#    of the file before firing `loadedmetadata`, issuing one small Range
#    request per fragment — pathologically slow on long videos (8h ≈ 3000
#    fragments → ~30 s and 100+ requests just to surface the first frame).
#    With sidx it's one Range request to fetch moov+sidx, then immediate
#    canplay.
#
# Output is still a single .mp4 byte-stream and plays in any modern
# browser / Plyr unmodified.
_FMP4_MOVFLAGS = '+frag_keyframe+empty_moov+default_base_moof+global_sidx'


def _is_fast_fmp4_cache(path: str) -> bool:
    """True iff `path` is a fragmented MP4 with a top-level ``sidx`` index.

    Both conditions are required for fast browser playback: ``moof`` makes
    it fragmented, ``sidx`` gives the browser a time→byte map so it can
    fire ``loadedmetadata`` after a single Range request. A file written
    before we adopted either flag is treated as a stale cache and rebuilt.
    Walks only top-level box headers — cost is O(top-level boxes),
    independent of file size.
    """
    has_moof = False
    has_sidx = False
    try:
        with open(path, 'rb') as f:
            file_size = os.fstat(f.fileno()).st_size
            offset = 0
            while offset + 8 <= file_size:
                f.seek(offset)
                header = f.read(8)
                if len(header) < 8:
                    return False
                size = int.from_bytes(header[:4], 'big')
                box_type = header[4:8]
                if box_type == b'moof':
                    has_moof = True
                elif box_type == b'sidx':
                    has_sidx = True
                if has_moof and has_sidx:
                    return True
                # Once we've seen the first moof we've passed the head
                # metadata — sidx must appear before the first fragment.
                if has_moof:
                    return False
                if size == 1:
                    # 64-bit extended size lives in the next 8 bytes.
                    ext = f.read(8)
                    if len(ext) < 8:
                        return False
                    size = int.from_bytes(ext, 'big')
                if size == 0:
                    # `mdat` (or any final box) with size 0 means "to EOF".
                    return False
                if size < 8:
                    return False
                offset += size
            return False
    except OSError:
        return False


def _drop_stale_cache(cache_path: str) -> None:
    """Delete `cache_path` if it isn't an fMP4 in the form we now produce.

    Lets the next ffmpeg invocation regenerate it. No-op when the cache is
    already fragmented + has sidx, or absent. Removal failures degrade to
    a warning — `_run_ffmpeg`'s atomic rename will overwrite the stale
    file on the rebuild anyway.
    """
    if not os.path.exists(cache_path):
        return
    if _is_fast_fmp4_cache(cache_path):
        return
    logger.info(f"Discarding stale cache (will rebuild as fMP4+sidx): {cache_path}")
    try:
        os.remove(cache_path)
    except OSError as e:
        logger.warning(f"Failed to remove stale cache {cache_path}: {e}")


def _nvenc_available() -> bool:
    """Whether ffmpeg has h264_nvenc available. Cached at first call."""
    global _nvenc_cache
    if _nvenc_cache is not None:
        return _nvenc_cache
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=10
        )
        _nvenc_cache = 'h264_nvenc' in (result.stdout or '')
        logger.info(f"NVENC {'detected' if _nvenc_cache else 'unavailable'} — "
                    f"transcode will use {'h264_nvenc' if _nvenc_cache else 'libx264'}")
    except Exception:
        _nvenc_cache = False
    return _nvenc_cache


def _run_ffmpeg(args: List[str], output_path: str, timeout: int = 600) -> bool:
    """Run ffmpeg writing to a temp file then atomically replace on success.

    `args` is the full ffmpeg argv EXCLUDING the trailing output path.
    Returns True on success (output file exists and is non-empty).
    """
    # Insert `.tmp` BEFORE the extension so ffmpeg can still infer the muxer
    # from the suffix (e.g. `.mp4`). Appending `.tmp` produces `...mp4.tmp`,
    # which ffmpeg rejects with "Unable to find a suitable output format".
    base, ext = os.path.splitext(output_path)
    tmp_path = base + '.tmp' + ext
    cmd = list(args) + [tmp_path]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='ignore')
            logger.error(f"ffmpeg exited with code {result.returncode}: {stderr[-500:]}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            logger.error("ffmpeg produced empty file")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
        # `os.replace` rather than `os.rename`: on Windows `rename` refuses
        # to clobber an existing destination, so a concurrent rebuild or a
        # `_drop_stale_cache` removal that lost a race would leave us
        # unable to commit the new output.
        os.replace(tmp_path, output_path)
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg timed out (>{timeout}s)")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
    except Exception as e:
        logger.error(f"ffmpeg failed: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def _ensure_remuxed(file_path: str) -> str:
    """Stream-copy the input into an MP4 container. Zero quality loss, fast.

    Returns the remuxed path on success or the original on failure.
    """
    remuxed = _get_remuxed_path(file_path)
    _drop_stale_cache(remuxed)
    if os.path.exists(remuxed):
        return remuxed

    logger.info(f"Remuxing {file_path} → {remuxed}")
    args = ['ffmpeg', '-i', file_path, '-c', 'copy', '-movflags', _FMP4_MOVFLAGS, '-y']
    # Remux is mostly bound by sequential read+write of the source bytes. Allow
    # 30 min so 100+ GB inputs on slower disks don't get killed mid-flight.
    if _run_ffmpeg(args, remuxed, timeout=30 * 60):
        logger.info(f"Remux complete: {remuxed}")
        return remuxed
    return file_path


def _ensure_h264(file_path: str) -> str:
    """Transcode to H.264 if needed. Uses NVENC when available, else libx264.

    Result is cached at `<base>.h264.mp4`. Returns the cached path on success
    or the original file_path on failure / no-op.
    """
    info = _get_video_info(file_path)
    codec = info['codec'] if info else None
    if not codec or codec not in _NEEDS_TRANSCODE_CODECS:
        return file_path

    transcoded = _get_transcoded_path(file_path)
    _drop_stale_cache(transcoded)
    if os.path.exists(transcoded):
        return transcoded

    if _nvenc_available():
        logger.info(f"Transcoding {file_path} ({codec}) → H.264 via NVENC")
        args = ['ffmpeg', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
                '-i', file_path, '-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23',
                '-c:a', 'aac', '-movflags', _FMP4_MOVFLAGS, '-y']
    else:
        logger.info(f"Transcoding {file_path} ({codec}) → H.264 via libx264")
        args = ['ffmpeg', '-i', file_path, '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-c:a', 'aac', '-movflags', _FMP4_MOVFLAGS, '-y']

    if _run_ffmpeg(args, transcoded, timeout=600):
        logger.info(f"Transcoding complete: {transcoded}")
        return transcoded
    return file_path


def _get_video_duration(file_path: str) -> Optional[float]:
    """Return total duration in seconds via ffprobe, or None if unavailable."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'json', file_path],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(result.stdout or '{}')
        dur = data.get('format', {}).get('duration')
        return float(dur) if dur is not None else None
    except Exception:
        return None


_FFMPEG_TIME_RE = re.compile(rb'time=(\d+):(\d+):(\d+(?:\.\d+)?)')


async def _stream_ffmpeg_progress(args: List[str], output_path: str, total_seconds: Optional[float]):
    """Run ffmpeg async, yielding (event_type, payload_dict) tuples.

    Atomic-rename pattern matches `_run_ffmpeg`: writes to <base>.tmp.<ext>,
    renames on success. Parses `time=HH:MM:SS.ms` lines from stderr to compute
    a percent against `total_seconds` (when known).

    Events:
      ('progress', {processed, total, percent})
      ('complete', {path, sizeBytes})
      ('error',    {message})
    """
    base, ext = os.path.splitext(output_path)
    tmp_path = base + '.tmp' + ext
    cmd = list(args) + [tmp_path]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )

    last_emit = 0.0
    last_processed: Optional[float] = None
    err_tail: List[bytes] = []
    try:
        assert proc.stderr is not None
        # ffmpeg uses '\r' between in-place progress updates rather than '\n'.
        # Read until either delimiter so we don't deadlock on a buffer that
        # never sees a newline.
        buf = bytearray()
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                if buf:
                    line = bytes(buf)
                    buf.clear()
                else:
                    break
            else:
                buf.extend(chunk)
                # Split into completed segments on \r or \n; keep the trailing partial.
                segments = re.split(rb'[\r\n]', bytes(buf))
                buf = bytearray(segments[-1])
                for seg in segments[:-1]:
                    if not seg:
                        continue
                    err_tail.append(seg)
                    if len(err_tail) > 30:
                        err_tail.pop(0)
                    m = _FFMPEG_TIME_RE.search(seg)
                    if m:
                        h, mi, s = m.groups()
                        processed = int(h) * 3600 + int(mi) * 60 + float(s)
                        last_processed = processed
                        # Throttle to ~5 Hz so the SSE stream stays light.
                        now = time.monotonic()
                        if now - last_emit >= 0.2:
                            last_emit = now
                            percent: Optional[float] = None
                            if total_seconds and total_seconds > 0:
                                percent = max(0.0, min(100.0, processed / total_seconds * 100.0))
                            yield ('progress', {
                                'processed': processed,
                                'total': total_seconds,
                                'percent': percent,
                            })
                continue

        ret = await proc.wait()
        if ret != 0:
            err = b'\n'.join(err_tail).decode('utf-8', errors='ignore')
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            yield ('error', {'message': err[-500:] or f'ffmpeg exited with code {ret}'})
            return

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            yield ('error', {'message': 'ffmpeg produced empty file'})
            return

        # `os.replace` rather than `os.rename`: see _run_ffmpeg for the
        # Windows clobber-on-rename rationale.
        os.replace(tmp_path, output_path)
        # Final 100% so the client UI can settle on a finished bar.
        yield ('progress', {
            'processed': total_seconds if total_seconds else last_processed,
            'total': total_seconds,
            'percent': 100.0 if total_seconds else None,
        })
        yield ('complete', {
            'path': output_path,
            'sizeBytes': os.path.getsize(output_path),
        })
    except asyncio.CancelledError:
        # Client disconnected — stop ffmpeg promptly so we don't waste CPU/disk.
        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        raise


def _resolve_abs_dir(dir_param: Optional[str]) -> str:
    """Resolve an absolute directory path. The dir param IS the absolute path."""
    if not dir_param:
        raise HTTPException(status_code=400, detail='Directory parameter required')
    real = os.path.abspath(dir_param)
    if not os.path.isdir(real):
        raise HTTPException(status_code=400, detail=f'Not a directory: {dir_param}')
    return real


def _resolve_upload_dir(dir_param: Optional[str]) -> str:
    """Resolve and create a directory for uploaded files."""
    if not dir_param:
        raise HTTPException(status_code=400, detail='Upload destination required')
    real = os.path.abspath(dir_param)
    try:
        os.makedirs(real, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to prepare upload directory: {e}')
    if not os.path.isdir(real):
        raise HTTPException(status_code=400, detail=f'Invalid upload directory: {dir_param}')
    return real


def _allocate_unique_filename(directory: str, filename: str) -> str:
    """Avoid overwriting existing videos by suffixing duplicates."""
    stem, ext = os.path.splitext(filename)
    candidate = filename
    counter = 1
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f'{stem}-{counter}{ext}'
        counter += 1
    return candidate


@app.get('/api/home-dir')
def get_home_dir():
    """Return the current user's home directory path."""
    return {"home": os.path.expanduser("~")}


def _resolve_tilde(path: str) -> str:
    """Expand ~ to the user's home directory."""
    if path.startswith('~'):
        return os.path.expanduser(path)
    return path


@app.get('/api/dirs')
def list_dirs(prefix: Optional[str] = ''):
    """List subdirectories matching a prefix for autocomplete. Prefix is an absolute path."""
    try:
        prefix = prefix or '/'
        prefix = _resolve_tilde(prefix)

        # Ensure it starts with /
        if not prefix.startswith('/'):
            prefix = '/' + prefix

        # If the exact directory exists, show its children even without a trailing slash.
        if not prefix.endswith('/') and os.path.isdir(prefix):
            parent = prefix
            partial = ''
        # Split into parent dir and partial name being typed
        elif prefix.endswith('/'):
            parent = prefix.rstrip('/')
            partial = ''
        else:
            parent = os.path.dirname(prefix)
            partial = os.path.basename(prefix)

        # Root case
        if not parent:
            parent = '/'

        if not os.path.isdir(parent):
            return {"dirs": []}

        try:
            entries = os.listdir(parent)
        except PermissionError:
            return {"dirs": []}

        partial_lower = partial.lower()
        dirs = []
        for entry in sorted(entries):
            if entry.startswith('.'):
                continue  # Skip hidden dirs
            full_path = os.path.join(parent, entry)
            if os.path.isdir(full_path) and entry.lower().startswith(partial_lower):
                dirs.append(os.path.join(parent, entry))
        return {"dirs": dirs[:20]}
    except Exception:
        return {"dirs": []}


@app.get('/api/ls')
def list_directory(
    path: Optional[str] = '~',
    extensions: Optional[str] = None,
):
    """List full contents of a directory for the folder browser.

    Returns all non-hidden entries: directories first, then files (if extensions given).
    """
    try:
        resolved = _resolve_tilde(path or '~')
        if not resolved.startswith('/'):
            resolved = '/' + resolved
        resolved = resolved.rstrip('/') or '/'

        if not os.path.isdir(resolved):
            return {"entries": [], "resolved": resolved}

        ext_set = None
        if extensions:
            ext_set = {e.strip().lower() for e in extensions.split(',') if e.strip()}

        try:
            raw_entries = os.listdir(resolved)
        except PermissionError:
            return {"entries": [], "resolved": resolved}

        dirs = []
        files = []
        for entry in sorted(raw_entries, key=str.lower):
            if entry.startswith('.'):
                continue
            full_path = os.path.join(resolved, entry)
            if os.path.isdir(full_path):
                dirs.append({"name": entry, "type": "dir"})
            elif ext_set:
                _, ext = os.path.splitext(entry)
                if ext.lower() in ext_set:
                    files.append({"name": entry, "type": "file"})

        entries = (dirs + files)[:100]
        return {"entries": entries, "resolved": resolved}
    except Exception:
        return {"entries": [], "resolved": path or '~'}


@app.get('/api/paths')
def list_paths(
    prefix: Optional[str] = '',
    extensions: Optional[str] = None,
):
    """Autocomplete both files and directories. For inference input picking.

    Args:
        prefix: Absolute path prefix being typed.
        extensions: Comma-separated extensions to include (e.g., '.mp4,.avi'). If None, dirs only.
    """
    try:
        prefix = prefix or '/'
        prefix = _resolve_tilde(prefix)
        if not prefix.startswith('/'):
            prefix = '/' + prefix

        ext_set = None
        if extensions:
            ext_set = {e.strip().lower() for e in extensions.split(',') if e.strip()}

        # Resolve parent and partial
        if not prefix.endswith('/') and os.path.isdir(prefix):
            parent = prefix
            partial = ''
        elif prefix.endswith('/'):
            parent = prefix.rstrip('/') or '/'
            partial = ''
        else:
            parent = os.path.dirname(prefix)
            partial = os.path.basename(prefix)

        if not parent:
            parent = '/'
        if not os.path.isdir(parent):
            return {"paths": []}

        try:
            entries = os.listdir(parent)
        except PermissionError:
            return {"paths": []}

        partial_lower = partial.lower()
        results = []
        for entry in sorted(entries):
            if entry.startswith('.'):
                continue
            full_path = os.path.join(parent, entry)
            if not entry.lower().startswith(partial_lower):
                continue
            if os.path.isdir(full_path):
                results.append({"path": full_path, "type": "dir"})
            elif ext_set:
                _, ext = os.path.splitext(entry)
                if ext.lower() in ext_set:
                    results.append({"path": full_path, "type": "file"})
        return {"paths": results[:30]}
    except Exception:
        return {"paths": []}


@app.get('/api/files')
def list_files(dir: Optional[str] = None):
    """List video files in the specified absolute directory."""
    try:
        target_dir = _resolve_abs_dir(dir)
        all_entries = os.listdir(target_dir)
        all_files = {f for f in all_entries if os.path.isfile(os.path.join(target_dir, f))}
        # Hide cache files we generated ourselves (including the in-progress
        # `.tmp` files written by `_run_ffmpeg` during atomic-rename) — users
        # only care about source videos.
        cache_suffixes = (
            '.h264.mp4', '.remux.mp4',
            '.h264.tmp.mp4', '.remux.tmp.mp4',
        )
        video_files = sorted(
            f for f in all_files
            if os.path.splitext(f)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
            and not f.lower().endswith(cache_suffixes)
        )
        # Check which videos have matching CSV files. We support multiple
        # CSVs per video (rater A vs rater B, draft vs final, …) by globbing
        # `{base}.csv` and `{base}_*.csv` in the same directory. The legacy
        # `hasCsv` boolean stays for older clients; `csvFiles` is the new
        # list, sorted with the canonical `{base}.csv` first.
        csv_set = {f for f in all_files if f.lower().endswith('.csv')}
        files_info = []
        for f in video_files:
            base = os.path.splitext(f)[0]
            full_path = os.path.join(target_dir, f)
            info = _get_video_info(full_path)
            codec = info['codec'] if info else None
            container = info['container'] if info else None
            is_h264 = codec is not None and codec not in _NEEDS_TRANSCODE_CODECS
            # `_is_fast_fmp4_cache` (not bare `os.path.exists`) so stale
            # caches written before we adopted `+global_sidx` show up as
            # uncached, matching `/probe`. Otherwise the frontend would
            # request `action=raw`, the backend would happily serve the
            # stale file, and the user would sit on the slow-buffering
            # path forever.
            transcoded_path = _get_transcoded_path(full_path)
            remuxed_path = _get_remuxed_path(full_path)
            has_cached_h264 = (
                os.path.exists(transcoded_path) and _is_fast_fmp4_cache(transcoded_path)
            )
            has_cached_remux = (
                os.path.exists(remuxed_path) and _is_fast_fmp4_cache(remuxed_path)
            )
            # Match either the canonical name or anything starting with
            # `{base}_` (so `movie1.csv` and `movie1_v2.csv` both belong to
            # the `movie1` video, but `movie10.csv` does not).
            csv_files = sorted(
                c for c in csv_set
                if c == base + '.csv' or c.startswith(base + '_')
            )
            # Surface canonical first when present.
            if (base + '.csv') in csv_files:
                csv_files = [base + '.csv'] + [c for c in csv_files if c != base + '.csv']
            files_info.append({
                "name": f,
                "hasCsv": len(csv_files) > 0,
                "csvFiles": csv_files,
                "codec": codec,
                "container": container,
                "isH264": is_h264,
                "hasCachedH264": has_cached_h264,
                "hasCachedRemux": has_cached_remux,
                # Phase 4 of docs/pts-based-frame-mapping.md — frontend
                # uses avgFrameRate to drive the editor's frame-step UI
                # and shows a VFR badge when isVfr is True.
                "rFrameRate": info.get('rFrameRate') if info else None,
                "avgFrameRate": info.get('avgFrameRate') if info else None,
                "isVfr": info.get('isVfr') if info else None,
            })
        return {"files": [fi["name"] for fi in files_info], "filesInfo": files_info}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/upload-videos')
async def upload_videos(
    destination: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload local video files and CSV companions into a server directory."""
    target_dir = _resolve_upload_dir(destination)
    uploaded = []
    video_files = []
    skipped = []

    if not files:
        raise HTTPException(status_code=400, detail='No files uploaded')

    if not any(
        os.path.splitext(os.path.basename(upload.filename or ''))[1].lower() in ALLOWED_VIDEO_EXTENSIONS
        for upload in files
    ):
        raise HTTPException(status_code=400, detail='No supported video files were uploaded')

    for upload in files:
        original_name = os.path.basename(upload.filename or '')
        ext = os.path.splitext(original_name)[1].lower()

        if not original_name:
            skipped.append({'reason': 'Unnamed file'})
            continue

        is_video = ext in ALLOWED_VIDEO_EXTENSIONS
        is_label = ext in ALLOWED_LABEL_EXTENSIONS
        if not is_video and not is_label:
            skipped.append({'name': original_name, 'reason': 'Unsupported file format'})
            await upload.close()
            continue

        saved_name = _allocate_unique_filename(target_dir, original_name)
        target_path = os.path.join(target_dir, saved_name)

        try:
            with open(target_path, 'wb') as out_file:
                while True:
                    chunk = await upload.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    out_file.write(chunk)
            uploaded.append({
                'originalName': original_name,
                'savedName': saved_name,
                'type': 'video' if is_video else 'csv',
            })
            if is_video:
                video_files.append(saved_name)
        except Exception as e:
            if os.path.exists(target_path):
                os.remove(target_path)
            raise HTTPException(status_code=500, detail=f'Failed to upload {original_name}: {e}')
        finally:
            await upload.close()

    if not video_files:
        raise HTTPException(status_code=400, detail='No supported video files were uploaded')

    return {
        'directory': target_dir,
        'files': video_files,
        'uploaded': uploaded,
        'skipped': skipped
    }


@app.get('/api/files/{filename}')
def get_file(
    filename: str,
    request: Request,
    dir: Optional[str] = None,
    action: Optional[str] = None,
    prefer_native: bool = False,
):
    """Stream or return a single file.

    The frontend chooses how the server should serve via `action`:
      - 'raw'       : serve the original (subject to a server-side codec+container allowlist)
      - 'remux'     : stream-copy into MP4 (codec OK, container wrong)
      - 'transcode' : full re-encode to H.264 (codec not browser-friendly)
      - None        : legacy auto-detect via _ensure_h264
    The deprecated `prefer_native=1` is recognized as an alias of action='raw'.
    """
    base_dir = _resolve_abs_dir(dir)
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail='File not found')

    if prefer_native and not action:
        action = 'raw'

    if action == 'raw':
        # Existing caches are always MP4 + H.264 — prefer them over the original.
        transcoded = _get_transcoded_path(file_path)
        remuxed = _get_remuxed_path(file_path)
        if os.path.exists(transcoded):
            serve_path = transcoded
        elif os.path.exists(remuxed):
            serve_path = remuxed
        else:
            info = _get_video_info(file_path)
            codec = info['codec'] if info else None
            container = info['container'] if info else None
            if codec in _RAW_CAPABLE_CODECS and container in _RAW_CAPABLE_CONTAINERS:
                serve_path = file_path
            else:
                logger.warning(
                    f"action=raw rejected for {filename} (codec={codec}, container={container}); "
                    f"falling back to transcode"
                )
                serve_path = _ensure_h264(file_path)
    elif action == 'remux':
        serve_path = _ensure_remuxed(file_path)
    elif action in (None, 'transcode'):
        serve_path = _ensure_h264(file_path)
    else:
        raise HTTPException(status_code=400, detail=f'Invalid action: {action}')

    file_size = os.path.getsize(serve_path)
    content_type = mimetypes.guess_type(serve_path)[0] or 'application/octet-stream'
    range_header = request.headers.get('range')

    if range_header:
        try:
            range_spec = range_header.replace('bytes=', '').strip()
            start_str, end_str = range_spec.split('-', 1)
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            start = max(0, start)
            end = min(end, file_size - 1)

            if start > end:
                raise HTTPException(status_code=416, detail='Range not satisfiable')

            content_length = end - start + 1

            def file_streamer():
                with open(serve_path, 'rb') as f:
                    f.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk = f.read(min(CHUNK_SIZE, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            headers = {
                'Content-Range': f'bytes {start}-{end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(content_length),
                'Content-Type': content_type,
                'Content-Disposition': 'inline',
            }
            return StreamingResponse(file_streamer(), status_code=206, headers=headers)
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Range request parse error: {e}")

    return FileResponse(serve_path, headers={'Accept-Ranges': 'bytes'})


class LabelRectPayload(BaseModel):
    behavior: str
    timestamp: float
    endTimestamp: float


class BehaviorMetaPayload(BaseModel):
    name: str
    shortcut: Optional[str] = None


class LabelSavePayload(BaseModel):
    videoPath: str
    labelRects: List[LabelRectPayload]
    # Optional list of (name, shortcut) bindings. When present we emit a
    # `# trace-meta:` comment line above the header so shortcuts round-trip
    # through the CSV. Older clients omit this and the CSV stays unchanged.
    behaviors: Optional[List[BehaviorMetaPayload]] = None


def _format_trace_meta_line(behaviors: List[BehaviorMetaPayload]) -> Optional[str]:
    """Mirrors trace-annotator's buildTraceMetaLine — keep both in sync."""
    from urllib.parse import quote
    pairs = [
        f"{quote(b.name, safe='')}:{quote(b.shortcut, safe='')}"
        for b in behaviors if b.shortcut
    ]
    if not pairs:
        return None
    return f"# trace-meta: behaviors={','.join(pairs)}"


def _resolve_csv_name(filename: str, csv_name: Optional[str]) -> str:
    """Validate a user-supplied CSV name belongs to the video and is safe.

    `csv_name` must be a bare filename (no path separators), end in `.csv`,
    and start with `{video_base}.csv` or `{video_base}_…`. Anything else
    raises 400 — we never let arbitrary paths reach disk.
    """
    base, _ = os.path.splitext(filename)
    if not csv_name:
        return base + '.csv'
    if '/' in csv_name or '\\' in csv_name or '..' in csv_name:
        raise HTTPException(status_code=400, detail='Invalid csvName')
    if not csv_name.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail='csvName must end with .csv')
    if csv_name != base + '.csv' and not csv_name.startswith(base + '_'):
        raise HTTPException(status_code=400, detail='csvName must belong to this video')
    return csv_name


@app.post('/api/files/{filename}/labels')
def save_labels(
    filename: str,
    payload: LabelSavePayload,
    dir: Optional[str] = None,
    csvName: Optional[str] = None,
):
    """Save labelRects payload as CSV next to the video file."""
    base_dir = _resolve_abs_dir(dir)
    csv_path = os.path.join(base_dir, _resolve_csv_name(filename, csvName))
    try:
        with open(csv_path, 'w', newline='') as f:
            meta_line = (
                _format_trace_meta_line(payload.behaviors)
                if payload.behaviors else None
            )
            if meta_line:
                # csv.writer would quote the leading `#`; write the comment
                # line directly so the CSVImporter sees a plain prefix.
                f.write(meta_line + '\n')
            writer = csv.writer(f)
            writer.writerow(['labelId', 'timestamp', 'endTimestamp'])
            for rect in payload.labelRects:
                writer.writerow([rect.behavior, rect.timestamp, rect.endTimestamp])
        return {'success': True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save labels: {e}')


@app.get('/api/files/{filename}/csv')
def get_csv(filename: str, dir: Optional[str] = None, csvName: Optional[str] = None):
    """Return CSV file corresponding to filename (or specific csvName)."""
    base_dir = _resolve_abs_dir(dir)
    csv_path = os.path.join(base_dir, _resolve_csv_name(filename, csvName))
    if not os.path.exists(csv_path) or not os.path.isfile(csv_path):
        raise HTTPException(status_code=404, detail='CSV file not found')
    return FileResponse(csv_path, media_type='text/csv')


@app.delete('/api/files/{filename}/csv')
def delete_csv(filename: str, dir: Optional[str] = None, csvName: Optional[str] = None):
    """Delete a CSV file corresponding to filename (or specific csvName)."""
    base_dir = _resolve_abs_dir(dir)
    resolved_name = _resolve_csv_name(filename, csvName)
    csv_path = os.path.join(base_dir, resolved_name)
    if not os.path.exists(csv_path) or not os.path.isfile(csv_path):
        raise HTTPException(status_code=404, detail='CSV file not found')
    try:
        os.remove(csv_path)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete CSV: {e}')
    return {'success': True, 'name': resolved_name}


@app.post('/api/files/{filename}/rename-csv')
def rename_csv(
    filename: str,
    dir: Optional[str] = None,
    oldName: Optional[str] = None,
    newName: Optional[str] = None,
):
    """Rename an annotation CSV that belongs to {filename}.

    Both names must pass `_resolve_csv_name` (bare `.csv`, no path traversal,
    must belong to this video's name family). The target must not already
    exist — we never silently overwrite a peer rater's file.
    """
    if not oldName or not newName:
        raise HTTPException(status_code=400, detail='oldName and newName are required')
    base_dir = _resolve_abs_dir(dir)
    old_path = os.path.join(base_dir, _resolve_csv_name(filename, oldName))
    new_path = os.path.join(base_dir, _resolve_csv_name(filename, newName))
    if old_path == new_path:
        return {'success': True, 'name': newName}
    if not os.path.isfile(old_path):
        raise HTTPException(status_code=404, detail='Source CSV not found')
    if os.path.exists(new_path):
        raise HTTPException(status_code=409, detail='A CSV with that name already exists')
    try:
        os.rename(old_path, new_path)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f'Failed to rename CSV: {e}')
    return {'success': True, 'name': newName}


@app.post('/api/files/{filename}/transcode')
def transcode_file(filename: str, dir: Optional[str] = None):
    """Transcode a video to H.264 synchronously and return the result."""
    base_dir = _resolve_abs_dir(dir)
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail='File not found')
    result_path = _ensure_h264(file_path)
    transcoded = result_path != file_path
    return {
        'transcoded': transcoded,
        'path': result_path,
    }


@app.get('/api/files/{filename}/process-stream')
async def process_stream(
    filename: str,
    dir: Optional[str] = None,
    action: str = 'remux',
):
    """SSE: drive a remux or transcode and stream progress events.

    Events emitted (one per `data:` line, JSON):
        {type: 'started',  total, codec, container}
        {type: 'progress', processed, total, percent}
        {type: 'complete', path, sizeBytes, cacheName}
        {type: 'error',    message}
        {type: 'noop',     reason}            # cache already exists, nothing to do

    Why SSE: ffmpeg can take minutes; the original sync `_ensure_remuxed` /
    `_ensure_h264` blocked the request handler with no progress signal back to
    the user. This endpoint streams `time=HH:MM:SS` updates parsed from
    ffmpeg's stderr at ~5 Hz.
    """
    if action not in ('remux', 'transcode'):
        raise HTTPException(status_code=400, detail=f'Invalid action: {action}')

    base_dir = _resolve_abs_dir(dir)
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail='File not found')

    info = _get_video_info(file_path)
    codec = info['codec'] if info else None
    container = info['container'] if info else None

    if action == 'remux':
        output_path = _get_remuxed_path(file_path)
        ffmpeg_args = ['ffmpeg', '-i', file_path, '-c', 'copy',
                       '-movflags', _FMP4_MOVFLAGS, '-y']
    else:  # transcode
        output_path = _get_transcoded_path(file_path)
        if _nvenc_available():
            ffmpeg_args = ['ffmpeg', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
                           '-i', file_path, '-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23',
                           '-c:a', 'aac', '-movflags', _FMP4_MOVFLAGS, '-y']
        else:
            ffmpeg_args = ['ffmpeg', '-i', file_path, '-c:v', 'libx264', '-preset', 'fast',
                           '-crf', '23', '-c:a', 'aac', '-movflags', _FMP4_MOVFLAGS, '-y']

    cache_name = os.path.basename(output_path)
    total_seconds = _get_video_duration(file_path)

    async def event_stream():
        # No-op fast path: cache already exists *and* is fragmented; otherwise
        # drop the stale cache and fall through to ffmpeg below.
        _drop_stale_cache(output_path)
        if os.path.exists(output_path):
            yield 'data: ' + json.dumps({
                'type': 'noop',
                'reason': 'cache exists',
                'path': output_path,
                'cacheName': cache_name,
                'sizeBytes': os.path.getsize(output_path),
            }) + '\n\n'
            return

        yield 'data: ' + json.dumps({
            'type': 'started',
            'action': action,
            'total': total_seconds,
            'codec': codec,
            'container': container,
            'cacheName': cache_name,
        }) + '\n\n'

        try:
            async for kind, payload in _stream_ffmpeg_progress(ffmpeg_args, output_path, total_seconds):
                if kind == 'complete':
                    payload = {**payload, 'cacheName': cache_name}
                yield 'data: ' + json.dumps({'type': kind, **payload}) + '\n\n'
        except asyncio.CancelledError:
            # Don't try to yield after cancellation; the stream is gone.
            raise
        except Exception as e:
            logger.exception(f"process-stream failed for {filename}")
            yield 'data: ' + json.dumps({'type': 'error', 'message': str(e)}) + '\n\n'

    return StreamingResponse(
        event_stream(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # disable proxy buffering for live updates
        },
    )


@app.get('/api/files/{filename}/probe')
def probe_file(filename: str, dir: Optional[str] = None):
    """Return codec + container info for a video file."""
    base_dir = _resolve_abs_dir(dir)
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail='File not found')
    info = _get_video_info(file_path)
    codec = info['codec'] if info else None
    container = info['container'] if info else None
    needs_transcode = codec in _NEEDS_TRANSCODE_CODECS if codec else False
    # A cache from before we adopted fMP4+sidx reads as "not cached" so
    # the frontend triggers process-stream and rebuilds it. The rebuild
    # path discards the stale file before invoking ffmpeg.
    transcoded_path = _get_transcoded_path(file_path)
    remuxed_path = _get_remuxed_path(file_path)
    return {
        'codec': codec,
        'container': container,
        'needsTranscode': needs_transcode,
        'transcodedCached': os.path.exists(transcoded_path) and _is_fast_fmp4_cache(transcoded_path),
        'remuxedCached': os.path.exists(remuxed_path) and _is_fast_fmp4_cache(remuxed_path),
        # Phase 4: same frame-rate metadata as `/api/files` exposes per
        # entry, so any single-file consumer (e.g. inference / probe UI)
        # can read avg fps + VFR flag without listing the whole dir.
        'rFrameRate': info.get('rFrameRate') if info else None,
        'avgFrameRate': info.get('avgFrameRate') if info else None,
        'isVfr': info.get('isVfr') if info else None,
    }


@app.get('/api/files/{filename}/pts')
def get_pts(filename: str, dir: Optional[str] = None):
    """Return per-frame presentation timestamps as a binary Float32 array.

    Used by the editor to snap behavior-interval boundaries to real frames
    on VFR sources (USB webcams), where ``time = frame / fps`` is wrong.
    First call may take a few seconds — decord reads the container index —
    and the result is cached to ``<video>.pts.npy`` next to the source for
    instant subsequent loads. Same code path the training/inference pipelines
    use; whichever side hits the video first builds the index for everyone.
    """
    base_dir = _resolve_abs_dir(dir)
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail='File not found')
    try:
        # Lazy: keep decord/numpy off the server's import surface for users
        # who never open a video (e.g. just browsing /api/dirs).
        from ..data_prep import _load_or_build_pts
        import numpy as np
        pts = _load_or_build_pts(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to build PTS index: {e}')
    # Float32 keeps ms precision (PTS rarely exceeds ~10⁵ s) and halves
    # bandwidth vs the float64 cache. Frame count goes in a header so the
    # client can sanity-check before allocating the typed array.
    payload = pts.astype(np.float32, copy=False).tobytes()
    return Response(
        content=payload,
        media_type='application/octet-stream',
        headers={
            'X-Frame-Count': str(len(pts)),
            'Cache-Control': 'private, max-age=86400',
        },
    )


@app.get('/alive')
def check_alive():
    return {"status": "alive"}


# ─── Bundled frontend static serving ───

def _find_static_dir():
    """Locate pre-built frontend assets inside the trace_tad package."""
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(pkg_root, "static", "annotator")
    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "index.html")):
        return candidate
    return None


# ── Jobs API router (must be included before SPA catch-all) ──
try:
    from trace_tad.server.jobs_router import router as jobs_router, manager as _job_manager
    app.include_router(jobs_router)

    @app.on_event("shutdown")
    async def _shutdown_jobs():
        _job_manager.cleanup_all()
except ImportError:
    pass  # trace_tad.jobs not installed


_static_dir = _find_static_dir()
if _static_dir:
    from fastapi.staticfiles import StaticFiles

    _assets_dir = os.path.join(_static_dir, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="static-assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the SPA frontend. All non-API routes fall through here."""
        file_path = os.path.join(_static_dir, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(_static_dir, "index.html"))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("trace_tad.server.app:app", host='0.0.0.0', port=8000, reload=True)
