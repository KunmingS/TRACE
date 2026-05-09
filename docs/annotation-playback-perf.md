# Annotation Playback Perf & Pause-Triggered Frame Cache

## Overview

Two related pieces of infrastructure in the annotator's editor view:

1. **An imperative playhead** that updates via `requestAnimationFrame` and a DOM
   `transform`, bypassing React's reconciler during playback so the timeline
   subtree (and every visible clip button) stops re-rendering at Plyr's
   `timeupdate` cadence.
2. **A pause-triggered frame cache** that pre-decodes ±30 frames around the
   current time into `ImageBitmap`s whenever playback pauses. Subsequent
   backward (and forward) frame steps paint instantly from the cache onto an
   overlay canvas while the real video catches up in the background.

The two work together: the imperative playhead means the timeline no longer
pays the per-tick React cost, and the frame cache means arrow-key stepping
inside the cached window is bounded by canvas paint time rather than codec
seek latency (historically ~50–100 ms per seek on HEVC).

## Why this exists

Previously, every Plyr `timeupdate` event (≈10 Hz with the existing throttle)
triggered a `setState({ currentTime })` on `Editor`. That value threaded down
as a plain prop through `EditorContainer → EditorBottomNavigationBar →
CustomTimeline → Playhead`. None of those components were `React.memo`-isolated
around a stable prop set, so every clip button in the timeline re-rendered on
each tick. With >20 behavior clips on screen the CPU hotspot was React
reconciliation, not video decode.

Separately, `stepFrame('backward')` simply wrote `playerRef.currentTime = t`.
Each arrow-left press was a fresh codec seek — visibly slow on long videos,
especially with transcoded H.264 content.

## Architecture

```
                     ┌──────────────────────────────────────┐
   Plyr timeupdate   │ Editor.handlePlyrTimeUpdate          │
   (~10 Hz)  ────────▶   playheadClock.current = t          │──► rAF in <Playhead>
                     │   throttledUpdate(setState, 250 ms)  │    reads clock, mutates
                     │                                      │    style.transform
                     └───────────────┬──────────────────────┘
                                     │
                                     ▼
                         setState @ 4 Hz — consumed only by:
                           • useAutoScroll  (CustomTimeline)
                           • ongoing-clip Math.max (CustomTimeline.tsx:121)
```

```
   Space (pause)                     ArrowLeft / ArrowRight
        │                                     │
        ▼                                     ▼
 FrameCache.prefetch(c)            Editor.stepFrame(dir)
   cancels in-flight                 targetFrame = floor(t·fps)
   abort, rebuilds                   ┌─── FrameCache.get(frame) ───┐
   cache = { frame → Bitmap }        │ hit                 miss     │
                                     ▼                     ▼
                            paintOverlayFrame()    frameCache.prefetch(frame)
                            (overlay canvas         (recenter window)
                             shows instantly)
                                     │
                                     ▼
                            playerRef.currentTime = t   (real seek)
                                     │
                                     ▼ Plyr `seeked`
                            handleRealVideoSeeked
                            if real frame == overlayDisplayedFrame:
                               hide overlay
```

## Imperative playhead

### The shared clock

`trace-annotator/src/views/EditorView/EditorBottomNavigationBar/playheadClock.ts`

```ts
export interface PlayheadClock { current: number; }
export const createPlayheadClock = (): PlayheadClock => ({current: 0});
```

A plain mutable ref object — no React state, no emitter. Owned by
`EditorContainer`:

```ts
const playheadClockRef = useRef<PlayheadClock>(createPlayheadClock());
// passed to both <Editor> and <EditorBottomNavigationBar>
```

### Writer — Editor

`Editor.handlePlyrTimeUpdate` writes to the clock on every Plyr `timeupdate`
without touching React state, and updates it synchronously from `stepFrame`,
`stepSeconds`, `jumpToTime`, and the `jumpToFrameIndex` branch of
`componentDidUpdate` via a `setClock(time)` helper. The companion
`setState({ currentTime })` path is throttled to **250 ms** (was 100 ms) and
kept only for `useAutoScroll` and the ongoing-clip width calculation.

### Reader — Playhead

`Playhead.tsx` attaches a `ref` to its root `<div>` and starts a
`requestAnimationFrame` loop inside a `useEffect`. Each tick reads
`playheadClock.current` and, if it changed, applies
`translate3d(${x}px, 0, 0) translateX(-50%)` to the element's
`style.transform`. React never re-renders the component on playback time
changes. `React.memo` still wraps the export for good measure.

The SCSS was updated accordingly: `transform: translateX(-50%)` on the root
is gone (owned by the inline transform now), and `filter: drop-shadow(...)`
on the triangle was removed because it would re-rasterize on every transform
update.

### What stays in React

Two downstream consumers still need `currentTime` as a prop:

- `useAutoScroll` in `CustomTimeline/useAutoScroll.ts` — runs an effect when
  `currentTime` changes to scroll the timeline container.
- `CustomTimeline.tsx:121` — `clipEnd = clip.isOngoing ? Math.max(currentTime,
  clip.end) : clip.end` extends the right edge of a currently-recording clip.

Both tolerate 4 Hz cadence with no visible loss.

## Playback speed

Plyr's built-in storage is disabled for the editor player. Its default
`localStorage` key is global (`plyr`) and includes `speed`, so selecting `0.5x`
once can make future annotation sessions start at half speed and look like
stutter. TRACE still exposes the speed menu for the current session, but each
new source is reset to `1x` in `Editor.resetPlaybackSpeed()`.

### What's explicitly not done (yet)

- `React.memo` on `CustomTimeline` / individual clip rows, `useCallback`
  stabilization of `onSelectClip`/`onRulerClick`, removal of always-on
  `will-change: transform` on `.ClipBar`, and narrower Redux subscriptions —
  deferred per scoping decision. Easy follow-up wins once the imperative
  playhead pass is validated.
- Replacing Plyr `timeupdate` with `requestVideoFrameCallback`. Not needed
  once the playhead is imperative; the coarse setState path is fine at
  250 ms.

## BehaviorShortcutsBar memoization

Old code called `isRecording(labelName)` once per label in the render loop,
and each call ran `imageData.labelRects.some(...)` — O(labels × labelRects)
per render. Replaced with:

```ts
const recordingLabelIds = useMemo(() => {
    const s = new Set<string>();
    for (const rect of imageData?.labelRects || []) {
        if (rect.labelId && !rect.endTimestamp) s.add(rect.labelId);
    }
    return s;
}, [imageData?.labelRects]);
// ...
const recording = recordingLabelIds.has(ln.id);
```

Plus `React.memo` on the default export.

## Frame cache

### Components

1. **`trace-annotator/src/logic/video/FrameCache.ts`** — new. The class:
   - holds a `Map<frameIndex, ImageBitmap>`;
   - owns an offscreen `HTMLCanvasElement` sized to the rendered video box
     (plus `devicePixelRatio`) and capped by a pixel budget;
   - tracks an `AbortController` for the in-flight prefetch.
2. **Hidden `<video>`** in `Editor.render()` (`className="cache-video"`,
   `display: none`, `preload="metadata"`). Same `src` as the main video;
   seeks here don't move the user-visible playhead.
3. **Overlay `<canvas>`** in `Editor.render()` (`className="frame-overlay"`,
   `position: absolute; inset: 0; z-index: 3; display: none`). `z-index: 3`
   sits one above `.plyr-video` (z-index 2). `pointer-events: none`.

### Lifecycle

| Event | Action |
|-------|--------|
| Plyr `loadedmetadata` | Clear old cache if any; construct `new FrameCache(cacheVideoEl, fps, { radius: 30, getRenderSize })` |
| `updateVideoSource`   | Set `cacheVideo.src`, call `.load()`, clear cache, hide overlay |
| Plyr `pause`          | `FrameCache.prefetch(Math.round(currentTime * fps))` |
| Plyr `play`           | `FrameCache.clear()` + hide overlay |
| `stepFrame`           | `FrameCache.get(target)` → `paintOverlayFrame()` (hit); else `prefetch(target)` to recenter |
| Plyr `seeked`         | If `round(realTime * fps) === overlayDisplayedFrame`, hide overlay |
| `jumpToTime` / `jumpToFrameIndex` | Hide overlay; if paused, `prefetch` around new location |
| `componentWillUnmount` | `FrameCache.clear()` |

### Prefetch algorithm

`prefetch(centerFrame)` aborts any in-flight prefetch, drops cached entries
outside the new window, builds an ordered target list that fans out from
the center frame (so the frames most likely to be stepped to first are
cached first), and awaits one seek at a time:

```
for each frame in order:
    if signal.aborted: return
    if cache.has(frame): continue
    await seekAndWait(hiddenVideo, frame/fps, 500 ms timeout, signal)
    size canvas to min(rendered video box × DPR, 2M pixels)
    ctx.drawImage(hiddenVideo, 0, 0, canvas.width, canvas.height)
    bitmap = await createImageBitmap(canvas)
    if signal.aborted: bitmap.close(); return
    cache.set(frame, bitmap)
```

`seekAndWait` resolves on the hidden video's `seeked` event, or on timeout,
or on abort. Sequential seeking matters: browsers throttle or drop
overlapping seeks on the same element.

### Why `createImageBitmap(canvas)` and not `OffscreenCanvas`

`OffscreenCanvas.transferToImageBitmap` is faster but has uneven browser
support. `createImageBitmap(HTMLCanvasElement)` is available in every
target browser (Chrome, Firefox, Safari) and is fast enough for 30–60
frames at typical video resolutions.

### Memory

Bounded at `2·radius + 1 = 61` bitmaps. Worst case at 1920×1080×4 B ≈
480 MB when the video is rendered at full 1080p and every cached frame hits
the 2M-pixel cap. Bitmaps are stored at rendered scale rather than source
scale, so 3840×1080 ultrawide inputs don't allocate native 4MP frames just to
paint an editor viewport.

```ts
ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
```

The overlay canvas still fills the video box via CSS, so frame-step feedback
remains instant while memory tracks the visible surface.

`clear()` calls `bitmap.close()` on every cached entry so GC releases
promptly; this runs on play, on source change, and on unmount.

### What this doesn't do

- The cache is **evicted on every play**. It does not persist across
  play/pause cycles — that was the approved scope. A sliding-window
  retain-during-play policy is possible but has ongoing memory cost.
- No fallback to single-video-element seeking. If the double `<video>` load
  ever measurably hurts on slow networks, the fallback is to seek the main
  video during pause and accept a visible flicker.

## Verification

### Dev server

From `/tank/skm/TRACE`:

```bash
trace app --dev
```

Load a video with ≥20 behavior clips.

### Playhead / timeline CPU

Open DevTools → Performance. Record 5 s of playback.

- **Before:** React commit phases for `CustomTimeline` / clip buttons fire at
  ~10 Hz.
- **After:** no React commits for the timeline subtree during steady-state
  playback (≤4 Hz from the throttled `setState`), and the playhead moves
  smoothly as rAF DOM-mutation tasks on a single element.

### BehaviorShortcutsBar

With ≥10 labels and an active recording, React Profiler should show a single
render for the bar rather than one render per label tick.

### Frame-cache backward step

1. Pause mid-clip. Wait ~1 s for prefetch to populate.
2. Press ArrowLeft repeatedly. Each step paints instantly — the overlay
   canvas appears, the Plyr video catches up within ~100 ms and the overlay
   hides on `seeked` match.
3. After 31 consecutive ArrowLeft presses from the initial pause point, the
   step falls outside the cached window and takes the slow path; the next
   prefetch re-centers.

### Eviction

Press Space to play. `FrameCache.clear()` runs; `performance.memory.usedJSHeapSize`
should drop.

### Regressions to sanity-check

- Forward playback still extends ongoing-clip width on the timeline (the
  Math.max path at `CustomTimeline.tsx:121`).
- Auto-scroll during playback still follows the playhead.
- Keyboard shortcuts (ArrowLeft/Right/Shift/Ctrl variants in
  `Editor.handleKeyDown`) unchanged.
- `jumpToTime` and `jumpToFrameIndex` (Redux) still seek; overlay hides on
  jump; playhead tracks via `playheadClock`.

### Leak check

DevTools → Memory → heap snapshots after 5 play/pause cycles. `ImageBitmap`
count should not grow — `bitmap.close()` in `FrameCache.clear()` releases
them.

## Files

**New:**
- `trace-annotator/src/logic/video/FrameCache.ts`
- `trace-annotator/src/views/EditorView/EditorBottomNavigationBar/playheadClock.ts`

**Modified:**
- `trace-annotator/src/views/EditorView/Editor/Editor.tsx`
- `trace-annotator/src/views/EditorView/Editor/Editor.scss`
- `trace-annotator/src/views/EditorView/EditorContainer/EditorContainer.tsx`
- `trace-annotator/src/views/EditorView/EditorBottomNavigationBar/EditorBottomNavigationBar.tsx`
- `trace-annotator/src/views/EditorView/EditorBottomNavigationBar/CustomTimeline/CustomTimeline.tsx`
- `trace-annotator/src/views/EditorView/EditorBottomNavigationBar/Playhead/Playhead.tsx`
- `trace-annotator/src/views/EditorView/EditorBottomNavigationBar/Playhead/Playhead.scss`
- `trace-annotator/src/views/EditorView/BehaviorShortcutsBar/BehaviorShortcutsBar.tsx`
