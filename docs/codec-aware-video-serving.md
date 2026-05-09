# Codec-Aware Video Serving

## Overview

TRACE serves videos through `GET /api/files/{filename}` and will automatically transcode codecs that most target browsers can't play. This document describes how the server decides when to transcode, how the client advises the server about browser capabilities, and how to extend the behavior.

Target browsers: **Windows Edge/Chrome, macOS Safari/Chrome**. The server runs on Linux.

## Why this exists

Previously every non-H.264 video was transcoded on first open inside the request handler. This caused two problems:

1. **Wasted work on codecs browsers already support.** AV1 and VP9 play natively on every target browser, but were transcoded anyway — doubling disk usage with a cached `.h264.mp4` and blocking the first request on an ffmpeg run.
2. **No way to let capable browsers skip transcoding.** Safari plays HEVC natively, but there was no signal from client to server saying "I can handle this — just send the original bytes."

The current design prunes the transcode list to codecs that actually need it, and lets the client opt out of transcoding for codecs that *some* browsers can play natively.

## Architecture

```
                       ┌─────────────────────────────────────────┐
    FileBrowser        │ build videoUrl:                         │
    (loads /api/files) │   base = /api/files/foo.mp4?dir=...     │
                       │   if codec=h264 or !needs_transcode     │
                       │     → base                              │
                       │   elif canPlayType(codec) == "probably" │
                       │     → base + &prefer_native=1           │
                       │   else                                  │
                       │     → base  (server will transcode)     │
                       └────────────────┬────────────────────────┘
                                        │ videoUrl in imageData
                                        ▼
                                Editor / Plyr source
                                        │ HTTP GET
                                        ▼
             ┌──────── trace_tad/server/app.py get_file() ───────────┐
             │ if prefer_native=1 AND codec in _MAYBE_NATIVE_CODECS: │
             │     serve original (skip _ensure_h264)                │
             │ else:                                                 │
             │     _ensure_h264(file_path) (sync transcode if needed)│
             └───────────────────────────────────────────────────────┘
```

### Codec classification

Two sets in `trace_tad/server/app.py` drive all decisions:

```python
_NEEDS_TRANSCODE_CODECS = {
    'hevc', 'h265',              # browser-dependent; honor prefer_native
    'mpeg4', 'mpeg2video',       # legacy; always transcode
    'msmpeg4v3', 'wmv3', 'flv1',
}

_MAYBE_NATIVE_CODECS = {'hevc', 'h265'}
```

| Codec | In transcode set? | In maybe-native set? | Effect |
|-------|-------------------|----------------------|--------|
| h264 | No | No | Always served directly. |
| av1 | No | No | Always served directly (all target browsers support it). |
| vp9 | No | No | Same. |
| hevc / h265 | Yes | Yes | Transcoded unless client sends `prefer_native=1`. |
| mpeg4, mpeg2video, msmpeg4v3, wmv3, flv1 | Yes | No | Always transcoded. `prefer_native` is ignored. |

### Server path (`trace_tad/server/app.py`)

```python
@app.get('/api/files/{filename}')
def get_file(filename, request, dir=None, prefer_native: bool = False):
    ...
    if prefer_native and _get_video_codec(file_path) in _MAYBE_NATIVE_CODECS:
        serve_path = file_path                    # skip transcode
    else:
        serve_path = _ensure_h264(file_path)      # transcode if in _NEEDS_TRANSCODE_CODECS
    ...
```

`prefer_native` is **advisory** — the server still runs ffprobe and only honors the flag for codecs in `_MAYBE_NATIVE_CODECS`. A confused or hostile client cannot force the server to serve a codec no browser handles.

`_ensure_h264` is unchanged: if the codec is in `_NEEDS_TRANSCODE_CODECS` and no cached `.h264.mp4` exists, ffmpeg runs synchronously and the output is cached on disk. The transcode is still sync — moving it onto the `JobManager` is a follow-up, not part of this change.

### Client path

Three pieces:

1. **`trace-annotator/src/utils/CodecSupport.ts`** — single source of truth for "does this browser play this codec?"

   ```ts
   export function browserCanPlayNatively(codec: string | undefined | null): boolean {
       if (!codec) return false;
       const c = codec.toLowerCase();
       const probe = (mime: string) =>
           document.createElement('video').canPlayType(mime) === 'probably';
       if (c === 'hevc' || c === 'h265') {
           return probe('video/mp4; codecs="hvc1"') || probe('video/mp4; codecs="hev1"');
       }
       if (c === 'h264' || c === 'avc1') return true;
       return false;
   }
   ```

   Only `"probably"` counts — `"maybe"` is too uncertain to risk an unplayable stream. Extend this function when adding support for a new codec (see "Adding a codec" below).

2. **`FileBrowser.tsx` — `isFilePlayable`** (determines whether clicking the item starts playback or blocks behind "Convert"):

   ```ts
   const isFilePlayable = (info) => {
       if (!info) return true;
       if (info.isH264 || info.hasCachedH264) return true;
       return browserCanPlayNatively(info.codec);
   };
   ```

3. **`FileBrowser.tsx` — `handleFileClick`** (appends `&prefer_native=1` when the browser can play the codec):

   ```ts
   const canPlayNative = info && !info.isH264 && !info.hasCachedH264
       && browserCanPlayNatively(info.codec);
   const nativeSuffix = canPlayNative ? '&prefer_native=1' : '';
   const videoUrl = `${API_URL}/api/files/${file}?dir=${...}${nativeSuffix}`;
   ```

### UI states

Badges rendered next to each file in `FileBrowser`:

| Badge | Condition | Meaning |
|-------|-----------|---------|
| `H264` | `isH264` or `hasCachedH264` | Browser-compatible H.264, streams directly. |
| `Native` | Non-H.264 + `browserCanPlayNatively(codec)` | Browser plays this codec natively, no conversion needed. |
| `Converting` (spinner) | User clicked Convert, ffmpeg running | Transient state for the manual conversion flow. |
| `Convert` (button) | Not playable and no cached H.264 | Click to trigger `POST /api/files/{filename}/transcode`. |

## Files touched

| File | Change |
|------|--------|
| `trace_tad/server/app.py` | Pruned `_NEEDS_TRANSCODE_CODECS`; added `_MAYBE_NATIVE_CODECS`; added `prefer_native` query param to `get_file()`. |
| `trace-annotator/src/utils/CodecSupport.ts` | New — `browserCanPlayNatively(codec)`. |
| `trace-annotator/src/views/EditorView/FileBrowser/FileBrowser.tsx` | Import helper; extended `isFilePlayable`; appended `prefer_native=1`; added "Native" badge branch. |
| `trace-annotator/src/utils/VideoUtil.ts` | Deleted dead `extractFrames()` that would OOM the tab on any real video. |
| `trace-annotator/src/views/MainView/TutorialPanel/TutorialPanel.tsx` | Updated step-01 HEVC copy. |

## Testing

Prepare a directory with `{h264.mp4, av1.mp4, vp9.mp4, hevc.mp4, legacy.wmv}` and point the file browser at it.

1. **Prune works** — AV1 and VP9 open immediately; no sibling `.h264.mp4` appears on disk after playing.
2. **HEVC native path (Safari)** — Network tab shows the request URL contains `&prefer_native=1`; response is 200 with range headers; disk unchanged.
3. **HEVC transcode path (Linux/Windows Chrome)** — URL has no `prefer_native`; first open blocks while ffmpeg runs; `.h264.mp4` appears on disk; subsequent opens are fast.
4. **Legacy codec ignores `prefer_native`** — even with the flag set, the server transcodes `wmv3` because it isn't in `_MAYBE_NATIVE_CODECS`.
5. **Badges render correctly** — `H264` for h264/av1/vp9; `Native` on Safari for HEVC; `Convert` on Chrome for HEVC and always for `.wmv`.
6. **Convert button still works** — clicking "Convert" fires `POST /api/files/{filename}/transcode` and updates the badge to `H264` on success.

## Adding a codec

To let the client skip transcoding for a new codec that some target browser supports natively:

1. In `trace_tad/server/app.py`, add the codec (lowercase, matching ffprobe output) to `_MAYBE_NATIVE_CODECS`. If it was previously always-transcoded, leave it in `_NEEDS_TRANSCODE_CODECS` so it still transcodes for clients that don't opt in.
2. In `trace-annotator/src/utils/CodecSupport.ts`, extend `browserCanPlayNatively` with the right `canPlayType()` MIME + codec string for the new codec. Test on every target browser — return `true` only for `"probably"`.
3. Nothing else changes: `isFilePlayable`, `handleFileClick`, and the badge logic all route through `browserCanPlayNatively` automatically.

To remove a codec from auto-transcoding entirely (like we did for AV1/VP9), just delete it from `_NEEDS_TRANSCODE_CODECS` — the server will serve the original bytes unconditionally. No client change needed.

## Out of scope / follow-ups

- **Async transcoding** — `_ensure_h264` still runs ffmpeg inside the request handler. On long/legacy videos this blocks the HTTP connection for minutes. The fix is to wrap the transcode as a `JobManager` job (see `trace_tad/jobs/manager.py`) and return a 202 with a job id, then have the editor subscribe to the existing SSE log stream until the cached file appears. Deferred because it touches `JobType` enum, adds a new runner, and wires SSE reconnect into `Editor.tsx`.
- **Transcode on upload** — `/api/upload-videos` could enqueue transcoding when a file in `_NEEDS_TRANSCODE_CODECS` lands. Becomes a one-line change once the async path above exists.
- **Client fallback on playback error** — if `canPlayType` says `"probably"` but the browser still fails (e.g. HEVC on a Windows Edge install without the HEVC extension), the Editor could retry with `?prefer_native=0`. Currently the user sees a generic error and can click the "Convert" button in FileBrowser manually.
