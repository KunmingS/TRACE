/**
 * Frontend-side video playback policy.
 *
 * The browser's `canPlayType` is the source of truth for what this user agent
 * can demux+decode. We probe each (codec, container) combination once at module
 * load and cache the results. The backend serves whatever the frontend asks for
 * via the `action` query param; this module decides what to ask for.
 */

export type Codec = 'h264' | 'hevc' | 'vp9' | 'av1';
export type Container = 'mp4' | 'webm' | 'matroska' | 'mov';
export type ServeAction = 'raw' | 'remux' | 'transcode';

export interface VideoFileInfo {
    codec?: string | null;
    container?: string | null;
    isH264?: boolean;
    hasCachedH264?: boolean;
    hasCachedRemux?: boolean;
}

const PROBE_MIME: Record<string, string> = {
    // Container 'mp4' or 'mov' both probe as video/mp4 — they share the ISO BMFF base.
    'h264_mp4':  'video/mp4; codecs="avc1.42E01E"',
    'h264_mov':  'video/mp4; codecs="avc1.42E01E"',
    'hevc_mp4':  'video/mp4; codecs="hvc1"',
    'hevc_mov':  'video/mp4; codecs="hvc1"',
    'vp9_mp4':   'video/mp4; codecs="vp09.00.10.08"',
    'vp9_webm':  'video/webm; codecs="vp9"',
    'av1_mp4':   'video/mp4; codecs="av01.0.00M.08"',
    'av1_webm':  'video/webm; codecs="av01.0.00M.08"',
    // matroska intentionally not listed — no major browser demuxes raw MKV.
};

function normalizeCodec(raw: string | undefined | null): Codec | null {
    if (!raw) return null;
    const c = raw.toLowerCase();
    if (c === 'h264' || c === 'avc1' || c === 'avc') return 'h264';
    if (c === 'hevc' || c === 'h265') return 'hevc';
    if (c === 'vp9') return 'vp9';
    if (c === 'av1') return 'av1';
    return null;
}

function normalizeContainer(raw: string | undefined | null): Container | null {
    if (!raw) return null;
    const c = raw.toLowerCase();
    if (c === 'mp4' || c === 'mov' || c === 'webm' || c === 'matroska') return c as Container;
    return null;
}

const _probeCache = new Map<string, boolean>();

function probe(key: string): boolean {
    const cached = _probeCache.get(key);
    if (cached !== undefined) return cached;
    const mime = PROBE_MIME[key];
    if (!mime) {
        _probeCache.set(key, false);
        return false;
    }
    const ok = typeof document !== 'undefined' &&
        document.createElement('video').canPlayType(mime) === 'probably';
    _probeCache.set(key, ok);
    return ok;
}

/** True if the browser can play this codec inside this container. */
export function browserCanPlay(
    codec: string | undefined | null,
    container: string | undefined | null,
): boolean {
    const c = normalizeCodec(codec);
    const ct = normalizeContainer(container);
    if (!c || !ct) return false;
    return probe(`${c}_${ct}`);
}

/**
 * Choose how the server should serve this file:
 *   'raw'       — stream as-is (cache or codec+container both browser-friendly)
 *   'remux'     — stream-copy into MP4 (codec OK, container wrong; e.g. H.264 in MKV)
 *   'transcode' — full re-encode to H.264 (codec needs converting)
 */
export function decideServeAction(info: VideoFileInfo): ServeAction {
    if (info.hasCachedH264 || info.hasCachedRemux) return 'raw';

    const codec = normalizeCodec(info.codec);
    if (!codec) return 'transcode';

    if (browserCanPlay(codec, info.container)) return 'raw';
    if (browserCanPlay(codec, 'mp4')) return 'remux';
    return 'transcode';
}

/**
 * Whether clicking a file should start playback (true) or block behind a
 * Convert button (false). Raw and remux are auto-handled; only full transcode
 * needs explicit user consent because of its long runtime.
 */
export function shouldAutoPlay(info: VideoFileInfo): boolean {
    return decideServeAction(info) !== 'transcode';
}

/**
 * Backward-compat shim for callers that only knew about codec.
 * @deprecated Use browserCanPlay(codec, container) instead.
 */
export function browserCanPlayNatively(codec: string | undefined | null): boolean {
    return browserCanPlay(codec, 'mp4');
}
