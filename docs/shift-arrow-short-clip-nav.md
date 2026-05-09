# Shift+Arrow Boundary Navigation on Short Behavior Clips

## Symptom

On the editor timeline, `Shift+→` is documented as "jump to next annotation
boundary". With a typical multi-second clip it worked. With a **short** clip
(one the user created by tapping the behavior shortcut key twice in quick
succession — duration on the order of a few frames), `Shift+→` from the start
of the clip either did nothing or visibly "skipped" the clip's end boundary
entirely, landing on the next clip's start instead.

The minimap still showed a small green sliver at the short clip's position, so
the clip existed; only the boundary-jump hot-key failed to land on it.

## Root cause

Two independent issues compound in `Editor.tsx#handleKeyDown`.

### 1. The playhead time used for navigation was a throttled snapshot

`handlePlyrTimeUpdate` feeds an imperative `PlayheadClock` every Plyr tick
(for the rAF-driven playhead — see `annotation-playback-perf.md`) but only
calls `setState({ currentTime })` at a **250 ms** throttle. React state is
deliberately lagged so the timeline subtree doesn't re-render at 10 Hz.

`handleKeyDown` then read `this.state.currentTime` to decide where to jump
from. At 30 fps, 250 ms is ~7.5 frames. So when the user pressed the
shortcut key to **create** a short clip, the `timestamp` /
`endTimestamp` it recorded came from the stale state — they could both
snap to the same stale value, producing a clip whose start and end parse to
(nearly) the same number. And when the user then pressed `Shift+→`, the
"from" time was again stale: the cursor was already past the short clip's
end, but the filter thought it was before the start.

### 2. Strict `>` / `<` ties on boundary filtering

The filter for the next boundary was

```ts
events.filter(e => e.time > currentTime);
```

If `currentTime` was numerically equal to (or within sub-frame floating-point
distance of) a boundary — which is exactly what (1) produces for a
short clip — that boundary was excluded. `Shift+→` either sat on the current
location with nothing to do, or silently skipped to the next clip.

## Fix

In `trace-annotator/src/views/EditorView/Editor/Editor.tsx`:

- Read the **live** player time inside `handleKeyDown`
  (`this.playerRef.currentTime`) rather than `this.state.currentTime`. The
  state snapshot is still fine for React rendering; it's only wrong as the
  source of truth for a keystroke that must agree with where the user sees
  the playhead right now.
- Apply a **half-frame epsilon** when filtering boundaries:
  `e.time < currentTime - eps` for the left filter, `e.time > currentTime + eps`
  for the right filter. Boundaries within a half-frame of the playhead count
  as "already passed", so `Shift+→` always advances.
- Extract the event collection into `collectBoundaryEvents()` and have it
  also read `rect.frame` / `rect.endFrame` as a fallback for rects that
  only carry frame indices (future-proofing — matches what the timeline
  already does when building clip bars in
  `EditorBottomNavigationBar.tsx#tracks`).

The same live-time value is now used by the shortcut-key toggle path
(`toggleBehaviorClip(...)`), so newly created short clips record with the
real playback time rather than the stale state value. New clips can no
longer collapse to zero duration from this source.

## Verifying the fix

1. `trace app --dev` (or rebuild with `trace dev build-frontend`).
2. Load any video and define a behavior with a single-key shortcut.
3. Play the video and tap the shortcut twice in rapid succession to create a
   short clip near `0s`. Then create a second, longer clip a second or two
   later.
4. Seek to `0s` (click the ruler or the clip). Press `Shift+→`. The playhead
   should land on the **end of the short clip**. Press `Shift+→` again —
   landing on the start of the second clip. Press again — landing on its end.
5. `Shift+←` in reverse order to confirm symmetry.

## Notes

- The 250 ms setState throttle is load-bearing for timeline perf and is not
  being changed. The fix sidesteps it by reading the clock source directly.
- `eps = 0.5 / frameRate` automatically narrows at higher frame rates.
  Since [`pts-based-frame-mapping.md`](pts-based-frame-mapping.md) shipped,
  `frameRate` is the file's actual avg fps from the backend (`/api/files`
  → `avgFrameRate`), not the previous hard-coded 30, so the window is
  correctly sized for 60 fps captures and webcam VFR alike.
- `Cmd/Ctrl+←/→` (which **moves** the nearest boundary to the current time
  rather than jumping the playhead) flows through the same helper and now
  also uses the live time — so "nudge this boundary to where I'm parked"
  records the real position.
