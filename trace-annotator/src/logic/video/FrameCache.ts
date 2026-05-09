// Pause-triggered frame cache.
//
// When the video is paused, we prefetch ±`windowRadius` frames around the
// current position by seeking a hidden <video> element and rasterizing each
// frame into a display-scaled `ImageBitmap`. On backward/forward step within
// the cached window, the Editor paints the bitmap onto an overlay canvas
// instantly instead of waiting for the main <video> to re-decode.
//
// The cache is evicted on play (see Editor.handlePlyrPlay) so memory is
// only held during pause.

const DEFAULT_SEEK_TIMEOUT_MS = 500;
const DEFAULT_MAX_BITMAP_PIXELS = 2_000_000;

interface RenderSize {
    width: number;
    height: number;
}

interface FrameCacheOptions {
    radius?: number;
    maxBitmapPixels?: number;
    getRenderSize?: () => RenderSize | null;
}

interface SeekResult {
    ok: boolean;
}

function seekAndWait(video: HTMLVideoElement, time: number, signal: AbortSignal, timeoutMs = DEFAULT_SEEK_TIMEOUT_MS): Promise<SeekResult> {
    return new Promise((resolve) => {
        if (signal.aborted) {
            resolve({ok: false});
            return;
        }

        const state: {
            done: boolean;
            timer: number | null;
            onSeeked: (() => void) | null;
            onAbort: (() => void) | null;
        } = {done: false, timer: null, onSeeked: null, onAbort: null};

        const finish = (ok: boolean) => {
            if (state.done) return;
            state.done = true;
            if (state.onSeeked) video.removeEventListener('seeked', state.onSeeked);
            if (state.onAbort) signal.removeEventListener('abort', state.onAbort);
            if (state.timer !== null) window.clearTimeout(state.timer);
            resolve({ok});
        };

        state.onSeeked = () => finish(true);
        state.onAbort = () => finish(false);

        video.addEventListener('seeked', state.onSeeked, {once: true});
        signal.addEventListener('abort', state.onAbort, {once: true});
        state.timer = window.setTimeout(() => finish(false), timeoutMs);

        try {
            // Guard against NaN / < 0 — some browsers throw.
            const clamped = Math.max(0, Math.min(video.duration || time, time));
            video.currentTime = clamped;
        } catch {
            finish(false);
        }
    });
}

export class FrameCache {
    private readonly video: HTMLVideoElement;
    private readonly fps: number;
    private readonly radius: number;
    private readonly maxBitmapPixels: number;
    private readonly getRenderSize?: () => RenderSize | null;
    private readonly cache = new Map<number, ImageBitmap>();
    private readonly canvas: HTMLCanvasElement;
    private readonly ctx: CanvasRenderingContext2D | null;
    private currentAbort: AbortController | null = null;

    constructor(video: HTMLVideoElement, fps: number, options: FrameCacheOptions = {}) {
        this.video = video;
        this.fps = fps;
        this.radius = options.radius ?? 30;
        this.maxBitmapPixels = options.maxBitmapPixels ?? DEFAULT_MAX_BITMAP_PIXELS;
        this.getRenderSize = options.getRenderSize;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
    }

    get(frame: number): ImageBitmap | null {
        return this.cache.get(frame) || null;
    }

    has(frame: number): boolean {
        return this.cache.has(frame);
    }

    getRadius(): number {
        return this.radius;
    }

    /**
     * Abort any in-flight prefetch and start a new one for the window
     * [center - radius, center + radius].
     *
     * Seeks the hidden cache video one frame at a time (browsers throttle
     * overlapping seeks). Already-cached frames are skipped. Frames falling
     * outside the new window are dropped so memory stays bounded.
     */
    prefetch(centerFrame: number): void {
        if (!this.ctx) return;

        // Cancel previous prefetch.
        if (this.currentAbort) {
            this.currentAbort.abort();
        }
        const abort = new AbortController();
        this.currentAbort = abort;
        const {signal} = abort;

        const first = Math.max(0, centerFrame - this.radius);
        const last = centerFrame + this.radius;

        // Drop entries outside the new window.
        for (const key of Array.from(this.cache.keys())) {
            if (key < first || key > last) {
                const bmp = this.cache.get(key);
                if (bmp) bmp.close?.();
                this.cache.delete(key);
            }
        }

        // Build an ordered prefetch list: fan out from the center so the
        // frames most likely to be stepped to first get cached first.
        const targets: number[] = [];
        targets.push(centerFrame);
        for (let d = 1; d <= this.radius; d++) {
            if (centerFrame - d >= 0) targets.push(centerFrame - d);
            targets.push(centerFrame + d);
        }

        // Lazily size the bitmap canvas once metadata is known. Cache frames
        // only at display scale (plus devicePixelRatio), capped by a pixel
        // budget, so ultrawide source videos don't allocate hundreds of MB.
        const sizeCanvas = () => {
            const w = this.video.videoWidth;
            const h = this.video.videoHeight;
            if (w <= 0 || h <= 0) return false;

            const renderSize = this.getRenderSize?.() ?? null;
            const dpr = window.devicePixelRatio || 1;
            const renderScale = renderSize && renderSize.width > 0 && renderSize.height > 0
                ? Math.min(
                    1,
                    (renderSize.width * dpr) / w,
                    (renderSize.height * dpr) / h
                )
                : 1;
            const pixelScale = this.maxBitmapPixels > 0
                ? Math.min(1, Math.sqrt(this.maxBitmapPixels / (w * h)))
                : 1;
            const scale = Math.min(renderScale, pixelScale);
            const targetWidth = Math.max(1, Math.round(w * scale));
            const targetHeight = Math.max(1, Math.round(h * scale));

            if (this.canvas.width !== targetWidth || this.canvas.height !== targetHeight) {
                this.canvas.width = targetWidth;
                this.canvas.height = targetHeight;
            }
            return true;
        };

        void (async () => {
            for (const frame of targets) {
                if (signal.aborted) return;
                if (this.cache.has(frame)) continue;

                const time = frame / this.fps;
                const result = await seekAndWait(this.video, time, signal);
                if (!result.ok || signal.aborted) return;

                if (!sizeCanvas()) continue;

                try {
                    this.ctx!.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
                    // createImageBitmap(canvas) is well-supported; avoids the
                    // OffscreenCanvas.transferToImageBitmap portability caveats.
                    const bmp = await createImageBitmap(this.canvas);
                    if (signal.aborted) {
                        bmp.close?.();
                        return;
                    }
                    this.cache.set(frame, bmp);
                } catch {
                    // Single-frame failure shouldn't abort the whole prefetch.
                }
            }
        })();
    }

    clear(): void {
        if (this.currentAbort) {
            this.currentAbort.abort();
            this.currentAbort = null;
        }
        for (const bmp of this.cache.values()) {
            bmp.close?.();
        }
        this.cache.clear();
    }
}
