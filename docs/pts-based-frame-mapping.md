# PTS-Based Timestamp ↔ Frame Mapping

**Status:** Shipped (Phases 1–4). See [What shipped](#what-shipped) below
for the as-built summary; the planning sections that follow it
(["Why"](#why), ["Design"](#design), ["Impact surface"](#impact-surface),
etc.) are preserved as historical context for the original design
decisions.
**Related docs:** [`lossless-video-pipeline.md`](lossless-video-pipeline.md),
[`codec-aware-video-serving.md`](codec-aware-video-serving.md).

## Why

TRACE today assumes **constant frame rate (CFR)** everywhere it converts
between seconds and frame indices. The assumption shows up in several places:

- `data_prep.py` reads every frame with `cv2.VideoCapture` (line 93–100) just to
  build a per-frame timestamp array, then `np.searchsorted`s the CSV
  annotations against it.
- `data_prep.py` writes `segment = [rel_start / fps, rel_end / fps]` (line 125)
  and `duration = frame / fps` (line 205), both derived from a single fps.
- `thumos.py` recovers fps as `frame / duration` (line 53, 121) and feeds it to
  the pipeline.
- `precision.py` uses a hard-coded `eval_fps = 30.0` (configs line 96–97) to
  convert predicted segments to frames.
- `tools/infer.py` computes `duration = num_frames / fps` (line 47–48).

For a true CFR file these are all internally consistent. For a **VFR file**
(USB webcams in dim labs are essentially always VFR — see
`lossless-video-pipeline.md` discussion on VFR in lab recordings), every one of
these conversions is wrong by an amount that depends on where in the video the
frame timing diverged from the average. The failure is silent: the model
trains on frames whose CSV annotation no longer matches the visual content,
and there is no exception, no warning, just degraded mAP.

The user's recording assumptions for TRACE are:
1. High-quality CFR cameras (lab-grade industrial / IP cameras), and
2. Ordinary USB webcams (which produce VFR by default).

We need a single mapping mechanism that is correct for both, with no
re-encoding and no quality loss on the original videos.

## Design

### Core idea

Each video gets a **PTS table** — a 1-D array of `float64` PTS values, one per
encoded frame, obtained directly from the container via
`decord.VideoReader.get_frame_timestamp(range(len(vr)))[:, 0]`. This is the
ground truth of "frame `i` is shown at time `pts[i]`" as recorded by the
encoder. It is correct for CFR (where `pts[i] = i / fps`) and for VFR (where
the spacing is irregular).

Every conversion between time and frame goes through this table:

- **timestamp → frame**: `np.searchsorted(pts, t)` (with an optional
  ±1-frame correction to pick the *nearest* PTS rather than the next one).
- **frame → timestamp**: `pts[i]` directly.

There is **no fps** anywhere in the conversion path. `fps` becomes a display-
only quantity (e.g. for the annotator's frame-step UI), and even there it
should come from `avg_frame_rate` reported by ffprobe, not from
`frame / duration`.

### Schema additions to `dataset.json`

Backwards-compatible — readers that don't know about the new fields fall back
to the existing `frame_segment` / `segment` / `duration` semantics.

```json
{
  "video1_clip_3": {
    "duration": 25.6,
    "frame": 768,
    "subset": "train",
    "annotations": [{
      "timestamp_sec": [3.413, 12.700],
      "frame_segment": [102, 381],
      "segment": [3.413, 12.700],
      "label": "grooming"
    }],
    "source_video": "/abs/path/to/video1.mp4",
    "source_frame_offset": 2304,
    "source_pts_table": "/abs/path/to/video1.pts.npy"
  }
}
```

New fields:
- `annotations[].timestamp_sec` — `[start_sec, end_sec]` from the CSV. The
  canonical source of truth. `segment` is now a derived alias (kept for
  backwards-compat readers).
- `source_pts_table` — absolute path to the cached `.npy` file containing the
  source video's full PTS array.

`frame_segment` is the **derived** representation, computed via
`np.searchsorted(pts_table, timestamp_sec)`. It is materialized at prep time
so training-time loaders don't need to do the lookup per `__getitem__`.

### PTS table caching

- Path: `<source_video>.pts.npy` next to the original.
- Invalidation: regenerate if the `.npy` is older than the source video's
  mtime, or if `len(decord.VideoReader(path)) != len(loaded_pts)`.
- Size: ≈ 8 bytes × frame count. A 4-hour 30 fps recording is ~3.4 MB.
  Negligible.
- Build cost: decord reads only the index, not the frames — seconds for hours
  of footage, vs minutes for the current `cv2.VideoCapture` per-frame loop.

### Behaviour for CFR vs VFR

The two paths are unified:

| | CFR @ 30fps | VFR webcam |
|---|---|---|
| `pts[i]` | `i / 30` (exact) | irregular, encoder-provided |
| `searchsorted(pts, t)` | `round(t × 30)` (within 1 frame) | exact nearest frame |
| Quantization error | ½ frame at source fps | ½ inter-frame gap (variable, but always ≤ ½ frame at the local instantaneous fps) |
| Re-encoding required | No | No |

Worst-case mapping error in either case is **half a frame at the local
fps** — well below the human reaction-time floor (~100 ms) that already
bounds annotation precision.

## Impact surface

Concrete file:line list of everything that needs to change. Compiled from a
full scan of the repo; this is the complete set.

### A. Backend — prep & schema

| File:line | Current | After |
|---|---|---|
| `trace_tad/data_prep.py:93–100` | cv2 per-frame `POS_MSEC` loop builds `ts_array` | `pts_array = _load_or_build_pts(video_path)` |
| `trace_tad/data_prep.py:105–106` | `np.searchsorted(ts_array, ...)` (cv2-derived) | Same call, but on decord PTS array |
| `trace_tad/data_prep.py:125` | `segment = [rel_start / fps, rel_end / fps]` | `segment = [pts[clip_start + rel_start] − pts[clip_start], …]` |
| `trace_tad/data_prep.py:163–164` | `duration_sec = actual_frames / fps` (ffmpeg fallback path) | `duration_sec = pts[clip_end] − pts[clip_start] + (1 / avg_fps)` |
| `trace_tad/data_prep.py:205` | `clips_data[i]["duration"] = frame / fps` | Same formula as above |
| `trace_tad/data_prep.py:298–307` | dataset.json writer | Add `timestamp_sec`, `source_pts_table` fields |
| **new** `trace_tad/data_prep.py` | — | `_load_or_build_pts(video_path) -> np.ndarray` helper with mtime invalidation |

### B. Backend — dataset loaders

| File:line | Current | After |
|---|---|---|
| `trace_tad/datasets/thumos.py:18–19` | Fallback `int(anno["segment"][0] / duration * frame)` | Prefer `frame_segment`; if absent, look up via PTS table |
| `trace_tad/datasets/thumos.py:53` | `fps = video_info["frame"] / video_info["duration"]` injected into pipeline | Inject `pts_table` reference (or path) instead; remove fps |
| `trace_tad/datasets/thumos.py:84–89` | `eval_fps`-based fallback in `ThumosPaddingDataset.get_gt()` | Use PTS table |
| `trace_tad/datasets/thumos.py:121` | Same fps injection as line 53 | Same change |
| `trace_tad/datasets/transforms/end_to_end.py:122–124` | Normalizes `gt_segments` by `duration` | Unchanged in spirit, but ensure `duration` is now the PTS-derived value |
| `trace_tad/datasets/transforms/end_to_end.py:268–270` | Same `duration`-based scaling | Same as above |
| `trace_tad/datasets/transforms/video_transforms.py:71–74` | `VideoInit` reads `avg_fps` and stores as `total_frames` selector | Keep avg_fps as display-only; primary work happens via frame indices already |

### C. Backend — eval & post-processing

| File:line | Current | After |
|---|---|---|
| `trace_tad/evaluations/precision.py:22–23, 38–39` | `gt_fps`, `eval_fps` constructor params (default 30.0) | Remove. Read PTS table per video instead |
| `trace_tad/evaluations/precision.py:74` | `num_frames = duration * eval_fps` | `num_frames = len(pts_table)` (or `video_info["frame"]`) |
| `trace_tad/evaluations/precision.py:92–93` | `start = floor(t0 * eval_fps)` | `start = np.searchsorted(pts_table, t0)` |
| `trace_tad/evaluations/precision.py:154–155` | `start_frame = int(seg[0] * eval_fps)` | Same `searchsorted` |
| `trace_tad/models/utils/post_processing/utils.py:52–65` | `convert_to_seconds` uses `snippet_stride * fps` | Use `pts_table[frame_idx]` to convert back to seconds |
| `configs/_base_/datasets/calms21.py:96–97` | `gt_fps=30.0`, `eval_fps=30.0` | Delete |

### D. Backend — inference

| File:line | Current | After |
|---|---|---|
| `tools/infer.py:42–49` | `probe_video` returns `(num_frames, fps, duration)` | Also build PTS table and cache as `<video>.pts.npy` |
| `tools/infer.py:70–93` | `generate_pseudo_annotations` writes pseudo dataset.json | Include `source_pts_table` field |
| `tools/infer.py:260–261` | Predictions already in seconds (via `convert_to_seconds`) | Unchanged once `convert_to_seconds` is PTS-based |

### E. Frontend

The CSV / annotation export path **does not need to change**. Timestamps are
already captured as seconds (`HTMLVideoElement.currentTime`) and written as
floats to the backend (`server/app.py:814–839`). Source of truth is already
seconds.

Two **separate, optional** frontend issues surfaced during the audit. They
are not on the critical path for PTS-based mapping and can be deferred:

| File:line | Issue | Suggested fix |
|---|---|---|
| `trace-annotator/src/views/EditorView/Editor/Editor.tsx:699` | Hard-coded `frameRate: 30` | Read `avg_frame_rate` from `/api/files` (needs backend addition); use for frame-step UI only |
| `trace-annotator/src/utils/BehaviorUtil.ts:27`, `Editor.tsx:539–541` | Timestamp captured as `currentTime` (wall-clock) during playback | Optional: `requestVideoFrameCallback` to snap to displayed frame's PTS. Not required for correctness once PTS-based mapping is in — `searchsorted` lands on the nearest real frame regardless. |
| `trace_tad/server/app.py:77–110` | `_get_video_info` doesn't expose fps | Add `r_frame_rate`, `avg_frame_rate`, `is_vfr` to `/api/files` response (enables the two items above and a VFR badge in FileBrowser) |

## Verification

Each phase below has a self-contained verification step. Run them in order.

### Phase-1 verification (prep)

```bash
# CFR consistency: PTS-based path must match fps-based path on a CFR file
python - <<'PY'
import numpy as np, decord
from trace_tad.data_prep import _load_or_build_pts  # new helper

vr = decord.VideoReader("/path/to/cfr_video.mp4")
pts = _load_or_build_pts("/path/to/cfr_video.mp4")
fps = vr.get_avg_fps()

# Random timestamps
ts = np.random.uniform(0, len(vr) / fps, size=200)
fps_idx = np.round(ts * fps).astype(int).clip(0, len(vr) - 1)
pts_idx = np.searchsorted(pts, ts).clip(0, len(vr) - 1)
print("max disagreement (frames):", np.max(np.abs(fps_idx - pts_idx)))
# Expected: ≤ 1 (sub-frame rounding)
PY
```

```bash
# VFR detection: show that a webcam file disagrees with fps-based mapping
python - <<'PY'
import numpy as np, decord
vr = decord.VideoReader("/path/to/webcam_vfr.mp4")
pts = vr.get_frame_timestamp(range(len(vr)))[:, 0]
fps = vr.get_avg_fps()
expected = np.arange(len(vr)) / fps
drift = pts - expected
print(f"max drift: {drift.max()*1000:.1f} ms, "
      f"std: {drift.std()*1000:.1f} ms")
# A clean CFR file: drift ≈ 0. A VFR webcam: tens of ms or more.
PY
```

### Phase-2 verification (training)

Run a full training on `data/dev_test/` (small 4-class set) before and after
the change. The dataset is CFR so mAP should be unchanged within run-to-run
variance.

```bash
trace train --model small --work-dir data/dev_test --pairs video.mp4=video.csv --nproc 1
# Compare best mAP to baseline
```

### Phase-3 verification (eval)

```bash
trace eval --model-dir data/dev_test/model_YYYYMMDD_HHMMSS
# Compare mAP@IoU breakdown to phase-2 run; should match
```

### Phase-4 verification (inference)

```bash
# Run inference on a known file; verify .pts.npy is created and reused
trace predict --model-dir data/dev_test/model_YYYYMMDD_HHMMSS --input /some/cfr.mp4
ls -la /some/cfr.mp4.pts.npy   # exists
trace predict --model-dir data/dev_test/model_YYYYMMDD_HHMMSS --input /some/cfr.mp4
# Second run should be faster (PTS table cached, not rebuilt)
```

### Cross-check: end-to-end on a VFR file

The real value of this change. Record a webcam clip in dim light (forces VFR
via auto-exposure), annotate a behavior in the browser at a precise visual
moment, and verify the model sees the *same* visual frame at training time.
Walk:

1. Pause webcam clip on a recognizable frame (e.g. mouse paw raised).
2. Note `videoElement.currentTime` displayed in the annotator (e.g. 4.137 s).
3. Save annotation.
4. After prep, open `dataset.json` and find the clip; read
   `frame_segment[0]`.
5. Open the source video with decord, call
   `vr.get_batch([source_frame_offset + frame_segment[0]]).asnumpy()`.
6. Save that frame as PNG and confirm by eye it shows the paw raised.

For a CFR file this works with or without the change. **Only after the change
does it work for VFR.**

## Migration

- **Existing prepared datasets** without `source_pts_table` keep working.
  Loaders fall back to `frame_segment` (which is correct for the original
  CFR-only assumption under which they were built) or to
  `segment / duration * frame` (the existing legacy path). No re-prep
  required for old CFR datasets.
- **`.pts.npy` cache files** are written next to source videos. They are
  small but should be added to `.gitignore` patterns where source videos
  live.
- **`gt_fps` / `eval_fps`** are removed from configs. Any user config
  overrides that set these will now warn-and-ignore in the eval constructor
  (one release of grace) before being deleted.

## Out of scope

- **Async PTS table construction** for very long videos (8+ hours). decord's
  index read is fast enough that synchronous build during prep is fine for
  every realistic lab recording.
- **Frontend VFR badge** in FileBrowser. Pure UX nicety — gated behind the
  optional Section E backend change.
- **`requestVideoFrameCallback` PTS snap** at annotation capture. Reduces
  the already-sub-frame error from ~16 ms (1 video tick) to 0. Not worth a
  PR on its own; bundle it with future frontend work.
- **Multi-stream / multi-track containers**. Out of scope; not a lab recording
  use case.

## Implementation phases & checklist

Track progress here. Each phase is independently mergeable.

### Phase 1 — Prep + schema (core)

- [x] Add `_load_or_build_pts(video_path)` helper in `trace_tad/data_prep.py`
- [x] Replace cv2 timestamp-array build (lines 93–100) with PTS table load
- [x] Replace `segment` / `duration` derivations to use PTS values
- [x] Extend `dataset.json` writer to include `timestamp_sec` and
  `source_pts_table`
- [x] Unit test: CFR file produces ≤1-frame disagreement between PTS path and
  legacy `round(t × fps)` path (sub-frame rounding only) —
  `tests/test_data_prep_pts.py::test_pts_searchsorted_matches_cfr_fps_within_one_frame`
- [x] Unit test: synthesized non-uniform PTS array (`monkeypatch`) yields the
  PTS-correct `frame_segment` rather than the avg-fps approximation —
  `tests/test_data_prep_pts.py::test_process_video_uses_pts_for_vfr_mapping`

### Phase 2 — Training/loader fps decoupling

Implementation note: Phase 2 was done as **additive PTS path + preserved
CFR fallback** rather than the originally-planned removal of fps. This
keeps legacy CFR datasets (e.g. `data/dev_test/`, prepped before the PTS
upgrade) working without re-prep — they hit the fallback exactly as
before. New PTS-aware datasets prefer the PTS path. The two paths
collapse to bit-equivalent results on a true CFR file.

- [x] `thumos.py`: inject `source_pts_table` alongside the existing
  `fps` (kept as legacy fallback for old datasets)
- [x] `formatting.py`: extend `Collect.meta_keys` default with
  `source_pts_table`, `source_frame_offset`, `clip_frame_count` so the
  PTS reference reaches `convert_to_seconds` via `meta`
- [x] `precision.py`: per-clip `clip_pts` cache + `searchsorted` on
  prediction binning (`_process_single_video_prediction`); `_load_clip_pts`
  helper handles missing/broken `source_pts_table` gracefully
- [x] `post_processing/utils.py`: `convert_to_seconds` PTS-aware via
  `np.interp` against the clip-relative PTS slice + per-process LRU
  cache (`_get_clip_pts`)
- [ ] ~~Remove `gt_fps` / `eval_fps` from
  `configs/_base_/datasets/calms21.py`~~ — kept as the legacy fallback
  knob; not removed
- [ ] ~~Backwards-compat shim warning~~ — not needed; the fallback is
  silent and well-tested
- [ ] Run `trace train --model small --work-dir data/dev_test --pairs video.mp4=video.csv` and
  verify mAP within ±1% of pre-change baseline (deferred — dev_test is
  legacy CFR and exercises only the fallback path; the PTS path is
  covered end-to-end by `tests/test_pts_phase2.py`)
- [x] Unit tests covering both paths on CFR + synthetic VFR + legacy
  fallback + missing/broken PTS file —
  `tests/test_pts_phase2.py` (9 cases)

### Phase 3 — Inference

- [x] `tools/infer.py:probe_video`: build + cache `.pts.npy`, return
  `(num_frames, duration, fps, pts_path)`; duration is now
  PTS-derived (`pts[-1] - pts[0] + 1/avg_fps`)
- [x] `generate_pseudo_annotations`: emit virtual-clip-shaped entries
  (`source_video` + `source_frame_offset=0` + `source_pts_table`) so
  the existing Phase 2 PTS plumbing in `convert_to_seconds` and
  `Precision` activates without further wiring; also fixes
  inference-on-non-mp4 (the container extension is now read from the
  source path, not the config-hard-coded `format="mp4"`)
- [x] Unit tests covering build/cache/reuse, `avg_fps == 0` fallback,
  pseudo-annotation schema, broken-video skip, and end-to-end loadability
  via `Precision._load_clip_pts` — `tests/test_pts_phase3.py` (7 cases)
- [ ] Verify `trace predict` end-to-end on a CFR clip; second run reuses
  cache (covered by unit tests; full end-to-end with a checkpoint is
  deferred until model artifacts are available in CI)

### Phase 4 — Frontend

- [x] Backend: `_get_video_info` returns `rFrameRate`, `avgFrameRate`,
  `isVfr` (parsed from ffprobe's `r_frame_rate` / `avg_frame_rate` rationals;
  `_parse_rational` helper handles `30/1`, NTSC `30000/1001`, single-number
  forms, and pathological inputs without raising)
- [x] Backend: `/api/files` and `/api/files/{filename}/probe` carry the
  new fields per file
- [x] Frontend: `ImageDataUtil.createImageDataFromUrl` takes an optional
  `frameRate` parameter; `FileBrowser.playFile` passes `info.avgFrameRate`
  (falling back to `rFrameRate`) into the `ImageData` it produces
- [x] Frontend: `Editor.tsx` reads `imageData.frameRate` at constructor
  time AND on Plyr's `loadedmetadata` event, and feeds the actual fps
  into `FrameCache` (instead of the hard-coded 30); the residual
  `state.frameRate || 30` defaults are kept as defensive fallbacks
  before the player has loaded
- [x] Frontend: VFR badge + actual-fps chip in `FileBrowser`'s file
  expansion panel; tooltip explains TRACE handles VFR via per-frame PTS
- [x] Tests: `_parse_rational` edge cases, `_get_video_info` on a CFR
  fixture, `/api/files` + `/probe` HTTP shape, ffprobe-failure
  null-fallback, `ImageDataUtil` Jest test for the new optional
  parameter — covered by `tests/test_backend_app.py` (5 new cases) and
  `trace-annotator/src/utils/__tests__/ImageDataUtil.test.ts` (3 cases)
- [ ] ~~`requestVideoFrameCallback` PTS snap~~ — explicitly out of
  scope (per the "Out of scope" section above); not required for
  correctness once PTS-based mapping is in

### Cross-cutting

- [ ] End-to-end VFR cross-check (the manual paw-raised test in
  Verification) — deferred until a VFR webcam recording is available.
  Synthetic-VFR coverage (`tests/test_data_prep_pts.py::test_process_video_uses_pts_for_vfr_mapping`,
  `tests/test_pts_phase2.py::test_convert_to_seconds_pts_diverges_from_cfr_on_vfr`,
  `…test_precision_uses_pts_for_prediction_binning`) exercises the same
  algebra against a non-uniform PTS array; the manual test only adds the
  visual sanity check.
- [x] `docs/`: this document now carries a "## What shipped"
  section (below) and the Status line at the top is updated, mirroring
  the `lossless-video-pipeline.md` style.

## What shipped

Final as-built summary across the four phases. The original plan above
is preserved unchanged (impact tables, design rationale, schema
example) so the design intent stays auditable; this section records
what is actually on disk and what to read first if you need to navigate
the code.

### Schema on disk

`dataset.json` entries written by `data_prep.py` now carry, in addition
to the pre-existing fields:

```json
{
  "vid_clip_3": {
    "duration": 25.6,
    "frame": 768,
    "subset": "train",
    "annotations": [{
      "label": "drinking",
      "frame_segment": [102, 381],
      "segment": [3.413, 12.700],
      "timestamp_sec": [3.413, 12.700]
    }],
    "source_video": "/abs/path/to/vid.mp4",
    "source_frame_offset": 2304,
    "source_pts_table": "/abs/path/to/vid.mp4.pts.npy"
  }
}
```

`source_pts_table` points at the per-source-video PTS cache,
`<video>.pts.npy`, written by `_load_or_build_pts` next to the source.
Cache invalidation is `mtime + length`. Per-clip seconds (`segment`,
`timestamp_sec`) are derived from PTS, not from `frame / fps`.

Old datasets without `source_pts_table` keep working unchanged through
a `fps`-based fallback path that is byte-identical to the historical
behaviour.

### Code map

| Surface | File | What it does |
|---|---|---|
| Helper | `trace_tad/data_prep.py:_load_or_build_pts` | Build/load `<video>.pts.npy` via `decord.VideoReader.get_frame_timestamp`. Mtime + length invalidation. |
| Prep | `trace_tad/data_prep.py:_process_video` | All `cv2 POS_MSEC` per-frame timestamp scanning is gone; PTS table replaces it. `segment` / `duration` derived from PTS. |
| Loader | `trace_tad/datasets/thumos.py` | `__getitem__` injects `source_pts_table` alongside the legacy `fps` so the pipeline carries it through. |
| Pipeline | `trace_tad/datasets/transforms/formatting.py` | `Collect.meta_keys` default extended with `source_pts_table` / `source_frame_offset` / `clip_frame_count`. |
| Post-proc | `trace_tad/models/utils/post_processing/utils.py:convert_to_seconds` | PTS-aware path uses `np.interp(frame_idx, arange(N), clip_pts)` for sub-frame seconds; per-process LRU cache (`_get_clip_pts`) avoids reloading on repeat windows. CFR fallback preserved. |
| Eval | `trace_tad/evaluations/precision.py` | `_load_clip_pts(video_info)` helper; `_import_ground_truth` precomputes `self.clip_pts[clip_name]` once; `_process_single_video_prediction` bins prediction-seconds via `searchsorted(clip_pts, t)` (CFR fallback when PTS is absent). |
| Inference | `tools/infer.py:probe_video` | Returns `(num_frames, duration, fps, pts_path)` and (re)builds the `.pts.npy` cache. |
| Inference | `tools/infer.py:generate_pseudo_annotations` | Emits virtual-clip-shaped entries with `source_video` + `source_frame_offset=0` + `source_pts_table`. Side benefit: inference now works on `.mkv` / `.avi` directly (the source extension is read from the path, not config). |
| Backend API | `trace_tad/server/app.py:_get_video_info` | ffprobe also reads `r_frame_rate` / `avg_frame_rate`; returns `rFrameRate`, `avgFrameRate`, `isVfr`. New helper `_parse_rational` handles `30/1` / NTSC / `0/0` / garbage uniformly. |
| Backend API | `trace_tad/server/app.py` (`/api/files`, `/api/files/{n}/probe`) | Each entry carries the three frame-rate fields per file. |
| Frontend | `trace-annotator/src/utils/ImageDataUtil.ts` | `createImageDataFromUrl` takes an optional `frameRate`; ignores non-positive values. |
| Frontend | `trace-annotator/src/views/EditorView/FileBrowser/FileBrowser.tsx` | `FileInfo` carries `rFrameRate` / `avgFrameRate` / `isVfr`; `playFile` threads `avgFrameRate` (with `rFrameRate` fallback) into `ImageData`; the file expansion panel renders an FpsChip and a VFR badge. |
| Frontend | `trace-annotator/src/views/EditorView/Editor/Editor.tsx` | Initial state and `loadedmetadata` event read `props.imageData.frameRate ?? 30`; `FrameCache` is constructed with the real fps. |

### Tests on disk

| File | Cases | Coverage |
|---|---:|---|
| `tests/test_data_prep_pts.py` | 7 | Phase 1: cache build/invalidate; CFR PTS-vs-fps within 1 frame; new schema fields in memory; synthetic VFR mapping correctness; multi-clip + boundary-crossing; `dataset.json` writer end-to-end; CSV-overflow clamp. |
| `tests/test_pts_phase2.py` | 9 | Phase 2: `convert_to_seconds` CFR↔PTS equivalence, VFR divergence, legacy fallback, source-frame-offset rebasing; `Precision` PTS-binned prediction, CFR fallback, `_load_clip_pts` failure modes; `ThumosSlidingDataset` sample injection (with + without virtual clips). |
| `tests/test_pts_phase3.py` | 7 | Phase 3: `probe_video` build/cache/reuse, `avg_fps == 0` fallback; `generate_pseudo_annotations` schema, bad-video skip, all-bad RuntimeError; Phase 3↔Phase 2 integration via `Precision._load_clip_pts`. |
| `tests/test_backend_app.py` | +5 (new) | Phase 4 backend: `_parse_rational` edge cases; `_get_video_info` on CFR fixture; `/api/files` and `/probe` HTTP shape; ffprobe-failure null fallback. |
| `trace-annotator/src/utils/__tests__/ImageDataUtil.test.ts` | 3 | Phase 4 frontend: `createImageDataFromUrl` carries `frameRate` only when positive. |

Total Python: **37 passed**. Frontend Jest: **3 new passed** (the
3 pre-existing failures in `MainView` / `PathPicker` /
`TutorialPanel` are unrelated and present on the baseline before this
work).

### Design choices that diverged from the plan

1. **`gt_fps` / `eval_fps` were not removed.** The plan called for
   deletion plus a warn-and-ignore shim. Instead, both stayed as the
   silent CFR fallback for legacy datasets without `source_pts_table`.
   On a true CFR file the PTS path collapses to division and the two
   are bit-equivalent, so the fallback is correct without nagging.
2. **No "what shipped to the loader" fps removal.** `thumos.py`
   still injects `fps = frame / duration` into the sample. It's no
   longer load-bearing for time ↔ frame math (PTS does that now), but
   removing it would break datasets that were prepped before this
   refactor. Treat it as a deprecated free pass.
3. **`requestVideoFrameCallback` PTS snap was deferred.** Listed as
   optional in the plan; the < 16 ms residual error it would shave is
   below the human reaction-time floor and below TRACE's per-frame
   resolution. Bundled with future frontend work if and when it
   becomes worth a PR.
4. **End-to-end VFR cross-check (paw-raised test) is deferred.**
   Synthetic-VFR coverage in three separate test files exercises the
   exact algebra; the visual sanity check will land the next time a
   real VFR webcam recording flows through the pipeline.
