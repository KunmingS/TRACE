import { API_URL } from '../../config';

// Per-frame presentation timestamps (seconds) for a source video, fetched
// from the server's `/api/files/{filename}/pts` endpoint. The backend builds
// `<video>.pts.npy` lazily on first request and caches it next to the source,
// so subsequent loads are instant. See docs/pts-based-frame-mapping.md.
//
// Float32 is enough: PTS values are seconds (rarely > 10⁵), and ms-precision
// is what the editor needs anyway. Using a typed array also avoids JSON
// parsing a million-element list for long videos.

export type PTSResult =
    | { status: 'done'; pts: Float32Array }
    | { status: 'fallback'; reason: string };

const inflight = new Map<string, Promise<PTSResult>>();
// Mirror of `inflight` populated when a promise settles so callers without
// async access (e.g. pointer-move handlers in trim-handle hooks) can still
// grab the array. ``loadPts`` is the only writer.
const resolved = new Map<string, PTSResult>();

const cacheKey = (filename: string, dir: string) => `${dir}::${filename}`;

export function loadPts(filename: string, dir: string): Promise<PTSResult> {
    const key = cacheKey(filename, dir);
    const existing = inflight.get(key);
    if (existing) return existing;
    const promise = fetchPts(filename, dir)
        .catch((err): PTSResult => ({
            status: 'fallback',
            reason: err instanceof Error ? err.message : String(err),
        }))
        .then(result => {
            resolved.set(key, result);
            return result;
        });
    inflight.set(key, promise);
    return promise;
}

// Returns the loaded PTS array, or null if the load hasn't resolved yet
// or fell back. Callers should treat null as "snap with nominal fps".
export function getPtsIfReady(filename: string, dir: string): Float32Array | null {
    const r = resolved.get(cacheKey(filename, dir));
    return r && r.status === 'done' ? r.pts : null;
}

async function fetchPts(filename: string, dir: string): Promise<PTSResult> {
    const url = `${API_URL}/api/files/${encodeURIComponent(filename)}/pts`
        + `?dir=${encodeURIComponent(dir)}`;
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error(`PTS endpoint returned ${res.status}`);
    }
    const buf = await res.arrayBuffer();
    // The server packs Float32 little-endian (numpy default on x86/ARM).
    // Browsers we target are also LE — no byte-swap path needed.
    const pts = new Float32Array(buf);
    if (pts.length === 0) {
        throw new Error('Empty PTS array');
    }
    return { status: 'done', pts };
}

// Drop a cached entry — used when the underlying file changes (rare).
export function invalidatePts(filename: string, dir: string): void {
    inflight.delete(cacheKey(filename, dir));
    resolved.delete(cacheKey(filename, dir));
}

// Find the index of the PTS entry closest to `t`. The PTS array is
// strictly monotonically increasing, so a plain binary search of the
// lower bound is enough — we then compare the two neighbors and return
// whichever is closer.
export function nearestPtsIndex(pts: Float32Array, t: number): number {
    if (pts.length === 0) return 0;
    let lo = 0;
    let hi = pts.length;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (pts[mid] < t) lo = mid + 1;
        else hi = mid;
    }
    if (lo === 0) return 0;
    if (lo === pts.length) return pts.length - 1;
    return (t - pts[lo - 1]) <= (pts[lo] - t) ? lo - 1 : lo;
}

// Largest index `i` such that `pts[i] <= t`. This is the frame the
// player is currently *showing* — frame N is on screen for the half-open
// interval ``[pts[N], pts[N+1])``. Used by frame-step so ±1 always
// advances by exactly one decoded frame, which matters on VFR sources
// where ``±1/fps`` straddles or skips frames.
export function floorPtsIndex(pts: Float32Array, t: number): number {
    if (pts.length === 0) return 0;
    if (t <= pts[0]) return 0;
    let lo = 0;
    let hi = pts.length;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (pts[mid] <= t) lo = mid + 1;
        else hi = mid;
    }
    return Math.max(0, lo - 1);
}

// Snap a time (seconds) to the nearest real frame. With a PTS array,
// returns the actual frame's presentation timestamp — correct for VFR
// sources where neighboring frames aren't 1/fps apart. Without one,
// rounds to the nearest nominal-fps grid line, which is what the older
// `time = frame / fps` math produced. The returned `time` is what
// should be persisted; `frame` is the canonical index for the same
// boundary.
export function snapTime(
    t: number,
    pts: Float32Array | null,
    fallbackFps: number,
): { time: number; frame: number } {
    if (pts && pts.length > 0) {
        const idx = nearestPtsIndex(pts, t);
        return { time: pts[idx], frame: idx };
    }
    const fps = fallbackFps > 0 ? fallbackFps : 30;
    const frame = Math.max(0, Math.round(t * fps));
    return { time: frame / fps, frame };
}
