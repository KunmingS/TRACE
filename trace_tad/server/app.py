"""TRACE annotation server — FastAPI application.

Handles video file serving, directory browsing, codec detection,
HEVC→H.264 transcoding, label management, and serves the bundled
frontend in production mode.
"""
import logging
import os
import subprocess
import json
from typing import Optional, List
import csv

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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

# Browser-incompatible codecs that need transcoding to H.264
_NEEDS_TRANSCODE_CODECS = {'hevc', 'h265', 'av1', 'vp9', 'mpeg4', 'mpeg2video', 'msmpeg4v3', 'wmv3', 'flv1'}


def _get_video_codec(file_path: str) -> Optional[str]:
    """Use ffprobe to detect the video codec of a file."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_name', '-of', 'json', file_path],
            capture_output=True, text=True, timeout=30
        )
        info = json.loads(result.stdout)
        streams = info.get('streams', [])
        if streams:
            return streams[0].get('codec_name', '').lower()
    except Exception as e:
        logger.warning(f"ffprobe failed for {file_path}: {e}")
    return None


def _get_transcoded_path(file_path: str) -> str:
    """Return path for the cached H.264 transcoded version."""
    base, _ = os.path.splitext(file_path)
    return base + '.h264.mp4'


def _ensure_h264(file_path: str) -> str:
    """If the video is not browser-compatible, transcode to H.264 and cache the result.
    Returns the path to serve (original or transcoded)."""
    codec = _get_video_codec(file_path)
    if not codec or codec not in _NEEDS_TRANSCODE_CODECS:
        return file_path

    transcoded = _get_transcoded_path(file_path)
    if os.path.exists(transcoded):
        # Already transcoded
        return transcoded

    logger.info(f"Transcoding {file_path} from {codec} to H.264 ...")
    try:
        result = subprocess.run(
            ['ffmpeg', '-i', file_path, '-c:v', 'libx264', '-preset', 'fast',
             '-crf', '23', '-c:a', 'aac', '-movflags', '+faststart',
             '-y', transcoded],
            capture_output=True, timeout=600
        )
        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='ignore')
            logger.error(f"ffmpeg exited with code {result.returncode}: {stderr[-500:]}")
            # Clean up partial file
            if os.path.exists(transcoded):
                os.remove(transcoded)
            return file_path
        if os.path.exists(transcoded) and os.path.getsize(transcoded) > 0:
            logger.info(f"Transcoding complete: {transcoded}")
            return transcoded
        else:
            logger.error("Transcoding produced empty file")
            return file_path
    except subprocess.TimeoutExpired:
        logger.error("Transcoding timed out (10 min limit)")
        if os.path.exists(transcoded):
            os.remove(transcoded)
        return file_path
    except Exception as e:
        logger.error(f"Transcoding failed: {e}")
        return file_path


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


@app.get('/api/pick-folder')
def pick_folder():
    """Open a native folder picker dialog and return the selected path."""
    import sys
    try:
        if sys.platform == 'darwin':
            # macOS: use osascript
            result = subprocess.run(
                ['osascript', '-e', 'POSIX path of (choose folder with prompt "Select video folder")'],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                return {"path": None}  # User cancelled
            path = result.stdout.strip().rstrip('/')
        else:
            # Linux/Windows: use tkinter
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askdirectory(title='Select video folder')
            root.destroy()
            if not path:
                return {"path": None}
        return {"path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        video_files = sorted(f for f in all_files if os.path.splitext(f)[1].lower() in ALLOWED_VIDEO_EXTENSIONS)
        # Check which videos have a matching CSV
        csv_set = {os.path.splitext(f)[0] for f in all_files if f.lower().endswith('.csv')}
        files_info = []
        for f in video_files:
            base = os.path.splitext(f)[0]
            full_path = os.path.join(target_dir, f)
            codec = _get_video_codec(full_path)
            is_h264 = codec is not None and codec not in _NEEDS_TRANSCODE_CODECS
            has_cached = os.path.exists(_get_transcoded_path(full_path))
            files_info.append({
                "name": f,
                "hasCsv": base in csv_set,
                "codec": codec,
                "isH264": is_h264,
                "hasCachedH264": has_cached,
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
    """Upload local video files into a server directory for annotation."""
    target_dir = _resolve_upload_dir(destination)
    uploaded = []
    skipped = []

    if not files:
        raise HTTPException(status_code=400, detail='No files uploaded')

    for upload in files:
        original_name = os.path.basename(upload.filename or '')
        ext = os.path.splitext(original_name)[1].lower()

        if not original_name:
            skipped.append({'reason': 'Unnamed file'})
            continue

        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            skipped.append({'name': original_name, 'reason': 'Unsupported video format'})
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
                'savedName': saved_name
            })
        except Exception as e:
            if os.path.exists(target_path):
                os.remove(target_path)
            raise HTTPException(status_code=500, detail=f'Failed to upload {original_name}: {e}')
        finally:
            await upload.close()

    if not uploaded:
        raise HTTPException(status_code=400, detail='No supported video files were uploaded')

    return {
        'directory': target_dir,
        'files': [item['savedName'] for item in uploaded],
        'uploaded': uploaded,
        'skipped': skipped
    }


@app.get('/api/files/{filename}')
def get_file(filename: str, request: Request, dir: Optional[str] = None):
    """Stream or return a single file. Automatically transcodes H.265/HEVC to H.264."""
    base_dir = _resolve_abs_dir(dir)
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail='File not found')

    # Transcode if needed (cached on disk after first request)
    serve_path = _ensure_h264(file_path)

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


class LabelSavePayload(BaseModel):
    videoPath: str
    labelRects: List[LabelRectPayload]


@app.post('/api/files/{filename}/labels')
def save_labels(filename: str, payload: LabelSavePayload, dir: Optional[str] = None):
    """Save labelRects payload as CSV next to the video file."""
    base_dir = _resolve_abs_dir(dir)
    base, _ = os.path.splitext(filename)
    csv_path = os.path.join(base_dir, base + '.csv')
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['labelId', 'timestamp', 'endTimestamp'])
            for rect in payload.labelRects:
                writer.writerow([rect.behavior, rect.timestamp, rect.endTimestamp])
        return {'success': True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save labels: {e}')


@app.get('/api/files/{filename}/csv')
def get_csv(filename: str, dir: Optional[str] = None):
    """Return CSV file corresponding to filename."""
    base_dir = _resolve_abs_dir(dir)
    base, _ = os.path.splitext(filename)
    csv_path = os.path.join(base_dir, base + '.csv')
    if not os.path.exists(csv_path) or not os.path.isfile(csv_path):
        raise HTTPException(status_code=404, detail='CSV file not found')
    return FileResponse(csv_path, media_type='text/csv')


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


@app.get('/api/files/{filename}/probe')
def probe_file(filename: str, dir: Optional[str] = None):
    """Return codec info for a video file."""
    base_dir = _resolve_abs_dir(dir)
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail='File not found')
    codec = _get_video_codec(file_path)
    needs_transcode = codec in _NEEDS_TRANSCODE_CODECS if codec else False
    transcoded_exists = os.path.exists(_get_transcoded_path(file_path))
    return {
        'codec': codec,
        'needsTranscode': needs_transcode,
        'transcodedCached': transcoded_exists,
    }


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
