# Lossless Video Pipeline (Server Remux + Training Virtual Clips)

Two related changes, both motivated by the same insight: TRACE was
re-encoding video in places where it didn't have to. The server transcoded
files whose codec the browser already understood (only the container was
wrong); training data prep re-encoded every clip at CRF 18 even though decord
can read frame ranges directly from the source. This doc describes the new
architecture for both.

This doc supersedes the relevant parts of
[`codec-aware-video-serving.md`](codec-aware-video-serving.md) — the
`prefer_native=1` flag and the `_MAYBE_NATIVE_CODECS` set described there are
gone, replaced with an explicit `action` query parameter chosen by the
frontend.

## Part 1 — Server-side: container-aware remux + GPU transcode + frontend policy

### Why

Three concrete bugs in the old design:

1. **H.264-in-MKV silently failed.** `_get_video_codec()` only detected the
   codec, so an MKV containing H.264 was reported as `isH264: true`. The badge
   read "H264", `isFilePlayable` returned true, and the click sent the raw MKV
   bytes to the browser — which can't demux MKV. The user saw an unplayable
   video with no UX path forward.
2. **`prefer_native=1` ignored container.** Safari plays HEVC, so for an
   `.hevc.mp4` file the frontend would set `prefer_native=1` and the server
   would skip transcoding. But for `.hevc.mkv`, Safari sets the same flag, and
   the server returned raw MKV bytes that Safari also couldn't demux.
3. **Always full re-encode.** Even when the codec was already browser-friendly
   and only the container was wrong, the server ran `libx264 -preset fast -crf
   23` — minutes of CPU work where a `-c copy` remux would have taken seconds.
   And on the GPU host, `libx264` (CPU) was the only encoder used, ignoring
   NVENC's 5–10× speedup.

### Architecture

The frontend's `canPlayType` is the source of truth for what this user agent
can demux+decode. The frontend probes once, picks an action per file, and the
backend is a thin executor with a server-side safety allowlist.

```
Frontend (CodecSupport.ts):
  Module-load probe map → { h264_mp4: 'probably', hevc_mp4: <browser>,
                            vp9_webm: 'probably', av1_webm: 'probably', ... }

Per file in FileBrowser:
  GET /api/files → { name, codec, container, hasCachedH264, hasCachedRemux }
  decideServeAction(info):
    cached_h264 || cached_remux → 'raw'
    browserCanPlay(codec, container) → 'raw'
    browserCanPlay(codec, 'mp4') → 'remux'
    else → 'transcode'

User clicks file:
  GET /api/files/{name}?dir=...&action={raw|remux|transcode}

Server (trace_tad/server/app.py):
  raw       → cached.h264.mp4 or cached.remux.mp4 if present;
               else original (validated against codec/container allowlist);
               else fall through to transcode
  remux     → ensure {original}.remux.mp4 (ffmpeg -c copy)
  transcode → ensure {original}.h264.mp4 (NVENC if available, libx264 fallback)
```

### Decision matrix

| File | codec | container | Cache state | Action | Badge |
|------|-------|-----------|-------------|--------|-------|
| `foo.mp4` | h264 | mp4 | none | `raw` | `Native` |
| `foo.webm` | vp9 | webm | none | `raw` | `Native` |
| `foo.mp4` | hevc | mp4 | none, **Safari** | `raw` | `Native` |
| `foo.mp4` | hevc | mp4 | none, **Chrome/Linux** | `transcode` | `Convert` |
| `foo.mkv` | h264 | matroska | none | `remux` | `Remux` (auto-play) |
| `foo.mkv` | hevc | matroska | none, **Safari** | `remux` | `Remux` (auto-play) |
| `foo.wmv` | wmv3 | avi | none | `transcode` | `Convert` |
| anything | — | — | `.h264.mp4` exists | `raw` (serves cache) | `H264` |
| anything | — | — | `.remux.mp4` exists | `raw` (serves cache) | `Remux` |

The `Remux` action auto-triggers on click (it's seconds, not minutes); only
`Convert` requires an explicit button click because of its long runtime.

### Server-side details

- **`_get_video_info(path)`** runs `ffprobe -show_entries
  stream=codec_name:format=format_name -of json` once per file, mtime-cached.
  Returns `{codec, container}`. Container disambiguation: ffprobe reports both
  `.mkv` and `.webm` as `format_name=matroska,webm` because WebM is a
  Matroska subset, so the file extension is used as the primary
  disambiguator.
- **`_run_ffmpeg(args, output_path, timeout)`** is the shared helper. It
  writes to a `<base>.tmp<ext>` file (the `.tmp` goes *before* the extension
  so ffmpeg can still infer the muxer from the suffix) and atomically renames
  on success. This fixes a pre-existing race where two parallel requests for
  the same uncached file could both spawn ffmpeg and corrupt the output.
- **Cache paths now keep the original extension**: `_get_transcoded_path` and
  `_get_remuxed_path` return `<original_full_name>.h264.mp4` and
  `<original_full_name>.remux.mp4` respectively. The previous scheme stripped
  the extension first, which meant `foo.mkv` and `foo.mp4` in the same
  directory shared a cache slot.
- **Cache files are hidden from `/api/files`**: the listing filter excludes
  any file ending in `.h264.mp4` or `.remux.mp4`.
- **Server safety net**: `action=raw` is honored only if the codec is in
  `_RAW_CAPABLE_CODECS = {h264, avc1, hevc, h265, vp9, av1}` and the
  container is in `_RAW_CAPABLE_CONTAINERS = {mp4, mov, webm}`. A buggy or
  hostile client can't trick the server into serving a codec/container
  combination no browser can read.
- **NVENC detection**: `_nvenc_available()` runs `ffmpeg -hide_banner
  -encoders` once at startup, caches the result. When available, transcode
  uses `-hwaccel cuda -c:v h264_nvenc -preset p4 -cq 23`.

### Frontend details

- **`browserCanPlay(codec, container)`** in `CodecSupport.ts` looks up the
  combination in a static MIME map and calls `canPlayType()`. Returns true
  only for `"probably"` (matching the previous policy). The probe result is
  module-cached.
- **`decideServeAction(info)`** is the single source of truth for the
  per-file decision. `FileBrowser.tsx` calls it once for each file in the
  list and once at click time.
- **Backward compat**: `prefer_native=1` is recognized as an alias for
  `action=raw`. Old clients still work.

### Files touched (Part 1)

| File | Change |
|------|--------|
| `trace_tad/server/app.py` | `_get_video_info` (codec+container, cached); `_normalize_container`; `_get_remuxed_path`; `_ensure_remuxed`; `_nvenc_available`; `_run_ffmpeg` shared helper with atomic rename; `action` query param dispatch in `get_file`; container/`hasCachedRemux` fields in `/api/files`; cache files filtered from listing. |
| `trace-annotator/src/utils/CodecSupport.ts` | Rewritten: `browserCanPlay(codec, container)`, `decideServeAction(info)`, `shouldAutoPlay(info)`. |
| `trace-annotator/src/views/EditorView/FileBrowser/FileBrowser.tsx` | `isFilePlayable` delegates to `shouldAutoPlay`; click builds URL with `action=...`; new `Remux` badge variant. |
| `trace-annotator/src/views/EditorView/FileBrowser/FileBrowser.scss` | Teal `.CodecBadge.remux` color variant. |

## Part 2 — Training: virtual clips + VideoReader LRU cache

### Why

`data_prep.py` chopped each source video into 768-frame `.mp4` clips with
`ffmpeg -c:v libx264 -preset fast -crf 18`. CRF 18 is "visually lossless" to
the human eye, but:

1. The model trains on subtly compressed frames while inference (`trace
   infer`) reads original videos directly via decord — a real but small
   train/inference distribution shift.
2. Disk usage roughly doubles (raw videos + clips).
3. Prep is slow (CPU-bound re-encode of every frame).

decord can already read arbitrary frame ranges from the source video at zero
quality cost. The clipping step exists only because the dataset loader
expects one file per clip. Lifting that assumption removes the re-encode
entirely.

### Architecture

```
Before:
  raw videos → ffmpeg -c:v libx264 -crf 18 → clip files (.mp4)
                                             dataset.json with {duration, frame, annotations}
  Training:  VideoInit opens clip file, total_frames = len(clip_reader)
             VideoDecode reads frame_inds from clip

After (virtual mode):
  raw videos → metadata only → dataset.json with
                                  {duration, frame, annotations,
                                   source_video, source_frame_offset}
  Training:  PrepareVideoInfo: filename = source_video
             VideoInit: reader from per-process LRU cache;
                        total_frames = clip_frame_count (logical clip span)
             VideoDecode: source_inds = clip_inds + source_frame_offset
```

### Schema change to `dataset.json`

Two new optional fields per clip entry:

```json
{
  "video1_clip_3": {
    "duration": 25.6,
    "frame": 768,
    "subset": "train",
    "annotations": [...],
    "source_video": "/abs/path/to/video1.mkv",
    "source_frame_offset": 2304
  }
}
```

Annotations still record `frame_segment` in clip-local coordinates (e.g.
`[0, 140]`). The translation to source-video coordinates happens once,
inside `VideoDecode`, by adding `source_frame_offset` to the frame indices.

When `source_video` is **absent** (legacy datasets), the loader falls back to
the old `data_path/<video_name>.mp4` lookup — existing prepared datasets
keep working unchanged.

### VideoReader LRU cache

Without caching, every `__getitem__` would reopen the source video and parse
its moov atom. With shuffled training over a dataset that spans many source
videos, that's a measurable per-sample overhead.

The cache is a simple `OrderedDict`-backed LRU keyed on absolute path,
defaulting to `maxsize=8`. Because each PyTorch `DataLoader` worker is a
separate process, each worker has its own cache. With
`persistent_workers=True` (already set in both `tridet_small.py` and
`tridet_large.py`), the cache persists across epochs and hot source videos
stay open.

For the typical TRACE dataset (a handful of long source videos),
`maxsize=8` means everything fits and the cache effectively makes source
opens free.

### Files touched (Part 2)

| File | Change |
|------|--------|
| `trace_tad/data_prep.py` | `_process_video()` and `prepare_dataset()` take `virtual_clips=True` (default). When virtual: skip the ffmpeg clipping block, record `source_video` (absolute path) + `source_frame_offset` (= `clip_idx * clip_frames`) instead. JSON writer carries the two new fields through. |
| `trace_tad/datasets/transforms/video_transforms.py` | `_VideoReaderCache` class + module-level instance; `VideoInit` uses cache, picks `total_frames` from `clip_frame_count` for virtual clips; `VideoDecode` adds `source_frame_offset` to flat indices before `get_batch`. |
| `trace_tad/datasets/transforms/end_to_end.py` | `PrepareVideoInfo` uses `source_video` directly when present. |
| `trace_tad/datasets/thumos.py` | Both `__getitem__` paths copy `source_video` / `source_frame_offset` / `clip_frame_count` from `video_info` into the pipeline sample dict. |
| `tools/prep_dataset.py` | New `--reencode-clips` flag (opt-in for legacy physical clipping). |
| `trace_tad/cli.py` | New `--reencode-clips` flag on `trace train` (forwarded to `prepare_dataset`). |

## Verification

### Server-side (run from project root)

```bash
# Smoke test the codec/container detection on a real MKV directory
python -c "
from fastapi.testclient import TestClient
from trace_tad.server.app import app
c = TestClient(app)
r = c.get('/api/files', params={'dir': '/tank/rui/c00228764'})
for fi in r.json()['filesInfo']:
    print(fi['name'], '→', fi['codec'], fi['container'])
"
# Expected: every file shows codec=h264, container=matroska
```

Browser-level recipe: `trace app`, paste an MKV directory into the file
browser, observe that each file shows a teal `Remux` badge, click it, and
confirm playback works (first click is a few seconds for the remux,
subsequent clicks are instant).

### Training-side (frame fidelity)

```bash
# Synthetic test: 60s video, 8 clips at 200 frames each
python - <<'PY'
from trace_tad.data_prep import prepare_dataset
import json, decord, numpy as np

model_dir, json_path, _ = prepare_dataset(
    '/path/to/test_dataset', clip_frames=200, virtual_clips=True
)
db = json.load(open(json_path))['database']
clip = next(v for v in db.values() if v.get('source_video'))

src = decord.VideoReader(clip['source_video'])
offset = clip['source_frame_offset']
# Pull frame 50 of the clip = source frame (offset + 50)
frame = src.get_batch([offset + 50]).asnumpy()[0]
print('frame shape:', frame.shape, '— byte-exact source frame, no re-encode loss')
PY
```

End-to-end transform pipeline: build a sample dict with `source_video` /
`source_frame_offset` / `clip_frame_count`, run `PrepareVideoInfo` →
`VideoInit` → `VideoDecode`, and verify `np.array_equal(pipeline_frames,
src.get_batch(offset + clip_inds))`. This was confirmed during development.

## Migration

- **Existing prepared datasets** (with physical clip files and no
  `source_video` field) keep working unchanged — the loader detects the
  absence of `source_video` and falls back to the legacy file lookup.
- **New `trace train --work-dir X --pairs video.mp4=video.csv` runs** default to virtual clips.
  Pass `--reencode-clips` to opt back into physical clipping (e.g. when
  shipping a self-contained dataset to a different machine).
- **Existing transcode/remux caches** in user directories are unaffected;
  they continue to be served. The path-naming change from
  `<base>.h264.mp4` to `<full_name>.h264.mp4` only applies to caches
  created after this change. Old caches remain valid because
  `_ensure_h264` returns the cache only if it exists at the *new* path; if
  not, it transcodes once more under the new naming. The cost is one
  re-transcode per affected file. Acceptable.

## Out of scope / Follow-ups

- **Async transcode** (deferred from the previous codec-aware doc): wrap
  `_ensure_h264` and `_ensure_remuxed` as `JobManager` jobs so the HTTP
  request returns 202 immediately and the editor subscribes to an SSE log
  stream until the cache appears. Worth it for HEVC transcode (minutes);
  remux is fast enough that sync is fine.
- **Native client** that bypasses the `<video>` element entirely (libmpv via
  Tauri) would remove the need for any container conversion. Considered and
  intentionally deferred — current server-side approach covers the
  common cases without the multi-platform packaging burden.
- **Lossless re-encode option** (`-crf 0` or FFV1) for users who *want*
  physical clip files but without quality loss. Easy to add as another
  `--reencode-clips` mode if it becomes useful.
