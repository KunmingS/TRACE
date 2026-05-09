# Home Page Mouse-Move Perf

## Overview

A set of small, visually-invariant CSS changes to `MainView` that remove the
per-hover layout cost on the two primary interactive regions of the landing
page (the left workspace nav and the intake cards). The entry animations and
hover look-and-feel are unchanged.

The fixes target **style recalc + layout cost during mouse movement over the
home page**, not paint throughput. They matter most on slower GPUs / forwarded
displays, where even a sub-millisecond main-thread stall per frame translates
to a visibly laggy cursor.

## Why this exists

A perf pass using Chrome's CDP `Performance.getMetrics` (driven via Playwright,
against a local `trace app` prod build) showed that sweeping the mouse over
the home-page cards triggered **16 layout invalidations in 4 seconds** — one
per hover-enter / hover-leave. The source was the hover rule on `.ModeButton`:

```scss
.ModeButton {
  transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
  &:hover { transform: translateY(-1px); border-color: rgba(0,0,0,0.16); }
}
```

`transform: translateY(-1px)` on a non-composited element forces Blink to
invalidate layout on the lift (and again on the drop), because the rendering
path has to reconcile the transformed bounds with the page flow.

Separately, the two full-viewport background layers (`.BgGrid`, `.BgGlow`) and
each nav tab had no paint-containment hint, so any invalidation inside them
(or inside overlapping siblings) could cascade to the surrounding layout tree.

None of this dropped frames on a fast Chromium rig — budget at 60 fps was
never exceeded — but the extra main-thread work is what the user perceives as
cursor drag on slower clients.

## Changes

Three targeted SCSS edits. All are visually invariant; the entry animations
(`brandIn`, `panelIn`, `surfaceIn`) and all `:hover` transitions still render
identically.

### 1. Composite the card lift (`ImagesDropZone.scss`)

```diff
 .ModeButton {
   ...
+  will-change: transform;
   transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
 }
```

`will-change: transform` promotes `.ModeButton` to its own compositor layer
**once**, when the element is painted. The hover `translateY(-1px)` then runs
on the compositor thread with zero layout/paint cost on the main thread.

### 2. Narrow `transition: all` on dropzone buttons (`ImagesDropZone.scss`)

```diff
 .LoadBtn, .GhostButton {
-  transition: all 0.15s ease;
+  transition:
+    background-color 0.15s ease,
+    border-color 0.15s ease,
+    box-shadow 0.15s ease,
+    color 0.15s ease,
+    opacity 0.15s ease;
 }
```

`transition: all` makes Blink evaluate transition interpolation for every
animatable property on every style change. Enumerating the five properties
these buttons actually animate (derived from their `:hover` + `:disabled`
rules) produces identical visuals with less style-recalc work.

### 3. Contain paint on background + nav (`MainView.scss`)

```diff
 .BgGrid   { ... pointer-events: none; contain: layout paint; z-index: 0; }
 .BgGlow   { ... pointer-events: none; contain: layout paint; z-index: 0; }
 .BgNoise  { ... pointer-events: none; contain: layout paint; z-index: 0; }

 .TabBtn {
   ...
+  contain: paint;
   transition: background 0.2s, color 0.2s, border-color 0.2s;
 }
```

`contain` is a rendering isolation hint. Blink will not consider the inside
of a contained element when computing layout / paint invalidation rects for
the outside, and vice versa. The three full-viewport background elements
have no descendants and don't affect each other, so `layout paint` is safe.
`.TabBtn` has only icon + text descendants, so `paint` is safe (no `layout`
because the button width is content-sized).

## What was *not* changed, and why

- The entry animations (`brandIn 0.6s`, `panelIn 0.55s`, `surfaceIn 0.26s`)
  are one-shot and only run on mount or route change. Their per-frame cost
  overlaps with the cursor path for <1 s after load; not worth touching.
- The remaining `transition: all 0.15s` rules in `TrainPanel.scss`,
  `TestPanel.scss`, `InferPanel.scss`, and `PipelineBuilder.scss` are behind
  the Pipeline tab, not the default landing view. Same pattern applies if we
  later want to do a second pass.
- No JS mousemove/pointermove listeners exist on `window`, `document`, or
  `#root`, so there is nothing to fix on the event side.

## Measurement method

- Chromium via Playwright MCP, 1440×900 viewport, prod build served by
  `trace app` on `:8000`.
- `CDPSession.send('Performance.getMetrics')` sampled before and after each
  scenario. Metrics used: `RecalcStyleDuration`, `RecalcStyleCount`,
  `LayoutDuration`, `LayoutCount`, `ScriptDuration`, `TaskDuration`.
- Each scenario: 4-second programmatic mouse sweep (`page.mouse.move`
  zig-zag) confined to the target region, preceded by a 0.5–1.5 s settle
  with the cursor parked at `(10, 10)`.

Regions:

| Scenario      | Bounding box (x, y)             |
| ------------- | ------------------------------- |
| Idle          | cursor at (10, 10), no motion   |
| Nav sweep     | `33–325, 263–379` (TabBtn area) |
| Card sweep    | `436–1360, 276–336` (ModeButton row) |

## Results

Before/after over the same 4-second sweep, identical input path:

**Card sweep (over `.ModeButton`)**

| Metric                | Before | After | Δ       |
| --------------------- | -----: | ----: | ------: |
| LayoutCount           |     16 |     0 | **−100%** |
| LayoutDuration (ms)   |    4.8 |   0.0 | **−100%** |
| RecalcStyleCount      |    196 |   193 |      ~0 |
| RecalcStyleDuration   |   34.8 |  35.7 |      ~0 |
| ScriptDuration (ms)   |   38.8 |  32.0 |   −17 % |
| TaskDuration (ms)     |    341 |   294 |   −14 % |

**Nav sweep (over `.TabBtn` column)**

| Metric                | Before | After | Δ    |
| --------------------- | -----: | ----: | ---: |
| RecalcStyleCount      |    295 |   304 |   ~0 |
| RecalcStyleDuration   |   45.3 |  46.4 |   ~0 |
| LayoutCount           |      0 |     0 |  —   |
| ScriptDuration (ms)   |   47.1 |  44.3 | −6 % |
| TaskDuration (ms)     |    361 |   360 |   ~0 |

**Idle** (post-settle, 4 s, cursor parked): 0 style recalc, 0 layout, 0 ms
script — unchanged.

### Reading the numbers

The card-sweep layout elimination is the headline: the compositor-promoted
lift means main-thread layout no longer fires when the cursor crosses a
card. The ~14 % drop in total task time is a real main-thread saving even
though no frames were being dropped before.

The nav-sweep numbers are essentially flat. That is expected: the nav
hover rule only changes `background` and `color`, neither of which triggers
layout. `contain: paint` on `.TabBtn` is a safety hint that prevents future
regressions (e.g., if someone adds `box-shadow` to the hover state, the
shadow's paint rect will stay inside the button).

### Caveat — measurement rig vs. user machine

These numbers are from a fast Chromium running locally against `trace app`.
The cursor drag that motivated this work was reported on a different
machine / display path; automation can drive CDP-synthetic mouse events but
cannot replicate the compositor + display pipeline on the user's setup.
The work done here is nevertheless unconditionally correct: layout
invalidations and wide `transition: all` rules cost main-thread time on
every client, and the changes are visually invariant.

## Files touched

- `trace-annotator/src/views/MainView/MainView.scss`
- `trace-annotator/src/views/MainView/ImagesDropZone/ImagesDropZone.scss`

After editing, run:

```bash
trace dev build-frontend
```

to rebuild the SPA bundle into `trace_tad/static/annotator/`, which is what
`trace app` (prod mode) ships.
