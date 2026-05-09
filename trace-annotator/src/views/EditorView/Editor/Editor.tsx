import React, { createRef, forwardRef } from 'react';
import './Editor.scss';
import {ImageData, LabelName, Subject} from '../../../store/labels/types';
import {AppState} from '../../../store';
import {connect} from 'react-redux';
import {updateImageDataById, updateActiveSubjectId} from '../../../store/labels/actionCreators';
import {ISize} from '../../../interfaces/ISize';
import { TimeUtil } from '../../../utils/TimeUtil';
import { toggleBehaviorClip } from '../../../utils/BehaviorUtil';
import { jumpToFrame, markVideoHasCsv } from '../../../store/general/actionCreators';
import { API_URL } from '../../../config';
import Plyr from 'plyr';
import 'plyr/dist/plyr.css';
import {PlayheadClock} from '../EditorBottomNavigationBar/playheadClock';
import {FrameCache} from '../../../logic/video/FrameCache';
import {floorPtsIndex, loadPts, snapTime} from '../../../logic/video/PTSCache';

interface IProps {
    size: ISize;
    imageData: ImageData;
    updateImageDataById: (id: string, newImageData: ImageData) => any;
    onVideoStateChange?: (state: {
        currentTime: number;
        duration: number;
        isPlaying: boolean;
        frameRate: number;
    }) => void;
    labelNames: LabelName[];
    subjects: Subject[];
    activeSubjectId: string | null;
    focusedLabelNameId: string | null;
    updateActiveSubjectId: (id: string | null) => any;
    jumpToFrameIndex?: number | null;
    jumpToFrame?: (frameIndex: number | null) => void;
    markVideoHasCsv: (filename: string) => any;
    playheadClock?: PlayheadClock;
}

// 'pending' = waiting on a request that hasn't started yet (rare — only the
// instant between source-change and the kick-off setState). 'active' = in
// flight. 'done' = finished. 'fallback' (frameIndex only) = fetch failed
// and the editor is reverting to nominal-fps math; surfaces as a small
// warning rather than blocking the UI.
type LoadStageStatus = 'pending' | 'active' | 'done' | 'fallback';

interface IState {
    duration: number;
    currentTime: number;
    isPlaying: boolean;
    videoUrl: string | null;
    frameRate: number;
    videoSourceLoaded: boolean;
    isBuffering: boolean;
    hasError: boolean;
    errorMessage: string;
    // First-load progress: video stream (B — metadata + initial playback
    // buffer, completes at Plyr's `canplay`) and per-frame PTS index (C).
    // Tracked separately so the loading overlay can show which step is
    // outstanding — C dominates wall-clock when the .pts.npy cache is
    // cold, B dominates over slow network. See docs/pts-based-frame-mapping.md.
    loadStages: {
        videoStream: LoadStageStatus;
        frameIndex: LoadStageStatus;
    };
}

class Editor extends React.Component<IProps, IState> {
    videoRef = createRef<HTMLVideoElement>();
    cacheVideoRef = createRef<HTMLVideoElement>();
    overlayCanvasRef = createRef<HTMLCanvasElement>();
    playerRef: Plyr | null = null;
    timeChangeId: string | null = null;
    private updateImageDataTimeout: number | null = null;
    private lastUpdateTime: number = 0;
    private saveTimeout: number | null = null;
    private lastSavedRectsJson: string = '';
    private frameCache: FrameCache | null = null;
    private overlayDisplayedFrame: number | null = null;
    // True while a seek we initiated (step / jumpToTime / jumpToFrameIndex)
    // is in flight. Lets the seeking/seeked handlers tell our own seeks apart
    // from external ones (e.g., user dragging Plyr's progress bar), so we can
    // keep the overlay covering the video for the former and reveal the live
    // video for the latter.
    private internalSeekInProgress: boolean = false;
    // True when the in-flight internal seek painted the overlay from a
    // FrameCache hit. On `seeked` we then trust that bitmap and skip
    // recapturing from the main <video>: the cache and main videos are
    // independent decoders, so a recapture swaps pixels even when the
    // logical frame is identical, which reads as a same-frame flicker.
    private overlayPaintedFromCacheForCurrentSeek: boolean = false;
    // Per-frame PTS array (seconds) — populated once C finishes. Held on
    // the instance rather than in Redux because Float32Array doesn't
    // serialize. Snap logic (PR2) will read this on every interval edit.
    private frameTimestamps: Float32Array | null = null;
    // Monotonically increasing token used to discard stale PTS responses
    // when the user swaps videos faster than the index can build.
    private ptsRequestSeq: number = 0;

    constructor(props) {
        super(props);
        // Frame rate comes from the backend's `/api/files` (Phase 4 of
        // docs/pts-based-frame-mapping.md). 30 is only a fallback for
        // unit tests / legacy callers that didn't populate the field.
        const initialFrameRate =
            (props.imageData?.frameRate && props.imageData.frameRate > 0)
                ? props.imageData.frameRate
                : 30;
        this.state = {
            duration: 0,
            currentTime: 0,
            isPlaying: false,
            videoUrl: null,
            frameRate: initialFrameRate,
            videoSourceLoaded: false,
            isBuffering: false,
            hasError: false,
            errorMessage: '',
            loadStages: { videoStream: 'pending', frameIndex: 'pending' },
        };
    }

    // Kick off both initial-load stages for the current source. B (video
    // stream) is started by Plyr the moment we assign `playerRef.source`,
    // so this method only fires C (PTS index) and resets stage status; the
    // canplay listener flips B to 'done' once the first frame is paintable.
    // Holding B until canplay (rather than durationchange) means the staged
    // overlay covers the whole "click-to-first-frame" gap, so the generic
    // Buffering… spinner is reserved for mid-playback stalls.
    // Stale-response guard via `ptsRequestSeq` keeps fast video-switching honest.
    private startInitialLoad = (filename: string | undefined, dir: string | undefined) => {
        this.frameTimestamps = null;
        this.setState({
            loadStages: { videoStream: 'active', frameIndex: 'active' },
        });
        if (!filename || !dir) {
            // Defensive fallback — without dir+filename we can't reach the
            // PTS endpoint, so degrade silently to nominal-fps math. Older
            // imageData (pre-FileBrowser stamping) hits this path.
            this.setState(prev => ({
                loadStages: { ...prev.loadStages, frameIndex: 'fallback' },
            }));
            return;
        }
        const seq = ++this.ptsRequestSeq;
        loadPts(filename, dir).then(result => {
            // User switched videos before this resolved — drop the result.
            if (seq !== this.ptsRequestSeq) return;
            if (result.status === 'done') {
                this.frameTimestamps = result.pts;
                this.setState(prev => ({
                    loadStages: { ...prev.loadStages, frameIndex: 'done' },
                }));
            } else {
                this.setState(prev => ({
                    loadStages: { ...prev.loadStages, frameIndex: 'fallback' },
                }));
            }
        });
    };

    private debouncedUpdateImageData = (imageData: ImageData) => {
        if (this.updateImageDataTimeout) {
            clearTimeout(this.updateImageDataTimeout);
        }

        this.updateImageDataTimeout = window.setTimeout(() => {
            const current = this.props.imageData;
            if (Math.abs(current.timestamp - imageData.timestamp) > 0.1 ||
                Math.abs(current.frameIndex - imageData.frameIndex) > 1) {
                this.props.updateImageDataById(imageData.id, imageData);
            }
            this.updateImageDataTimeout = null;
        }, 200);
    };

    private throttledUpdate = (callback: () => void, delay: number = 16) => {
        const now = Date.now();
        if (now - this.lastUpdateTime >= delay) {
            callback();
            this.lastUpdateTime = now;
        }
    };

    private setClock = (time: number) => {
        if (this.props.playheadClock) {
            this.props.playheadClock.current = time;
        }
    };

    private debouncedSaveLabels = () => {
        if (this.saveTimeout) {
            clearTimeout(this.saveTimeout);
        }
        this.saveTimeout = window.setTimeout(() => {
            this.saveLabelsToBackend();
            this.saveTimeout = null;
        }, 1000);
    };

    private saveLabelsToBackend = () => {
        const rectsToSave = this.props.imageData.labelRects.filter(rect =>
            rect.timestamp != null && rect.endTimestamp != null
        );
        if (rectsToSave.length === 0) return;

        // Persist behavior→shortcut bindings alongside the rects so the
        // backend can emit a `# trace-meta:` line at the top of the CSV.
        // Only labels with a shortcut are sent; the metadata line is omitted
        // entirely when no behavior has a binding (matches old CSV layout).
        const behaviors = (this.props.labelNames || [])
            .filter(ln => !!ln.shortcut)
            .map(ln => ({ name: ln.name, shortcut: ln.shortcut }));

        // Dedup key includes behaviors so that updating only a shortcut (no
        // clip changes) still triggers a save and the CSV reflects it.
        const rectsJson = JSON.stringify({
            rects: rectsToSave.map(r => ({
                behavior: r.behavior,
                timestamp: TimeUtil.parseTimestamp(r.timestamp!),
                endTimestamp: TimeUtil.parseTimestamp(r.endTimestamp!)
            })),
            behaviors,
        });

        // Skip if nothing changed
        if (rectsJson === this.lastSavedRectsJson) return;

        const dir = (this.props.imageData as any).videoPath;
        if (!dir) return;
        const filename = this.props.imageData.fileData.name;
        // The sidebar stamps `csvName` on imageData only after the user picks
        // or creates a CSV. Without that explicit target, do not fall back to
        // `{base}.csv`; a media-only click should not write annotations.
        const csvName = (this.props.imageData as any).csvName as string | undefined;
        if (!csvName) return;

        fetch(
            `${API_URL}/api/files/${filename}/labels?dir=${encodeURIComponent(dir)}&csvName=${encodeURIComponent(csvName)}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    videoPath: dir,
                    labelRects: rectsToSave.map(r => ({
                        behavior: r.behavior,
                        timestamp: TimeUtil.parseTimestamp(r.timestamp!),
                        endTimestamp: TimeUtil.parseTimestamp(r.endTimestamp!)
                    })),
                    behaviors
                })
            }
        )
        .then(res => {
            if (res.ok) {
                this.lastSavedRectsJson = rectsJson;
                // Tell the FileBrowser sidebar that the CSV now exists on
                // disk — its `filesInfo.hasCsv` snapshot from the last
                // /api/files scan is stale until the user reloads the dir.
                this.props.markVideoHasCsv(filename);
            }
        })
        .catch(() => {});
    };

    componentDidMount() {
        if (this.props.imageData.videoUrl) {
            this.setState({ videoUrl: this.props.imageData.videoUrl });
            const filename = this.props.imageData.fileData?.name;
            const dir = (this.props.imageData as any).videoPath as string | undefined;
            this.startInitialLoad(filename, dir);
        }
        document.addEventListener('keydown', this.handleKeyDown);
        this.initializePlayer();
    }

    componentDidUpdate(prevProps: IProps, prevState: IState) {
        const prevImageData = prevProps.imageData;
        const currentImageData = this.props.imageData;

        if (prevImageData.id !== currentImageData.id ||
            prevImageData.videoUrl !== currentImageData.videoUrl) {
            this.setState({
                videoSourceLoaded: false,
                isBuffering: true,
                hasError: false,
                videoUrl: currentImageData.videoUrl
            });
            if (currentImageData.videoUrl) {
                this.updateVideoSource(currentImageData.videoUrl);
                const filename = currentImageData.fileData?.name;
                const dir = (currentImageData as any).videoPath as string | undefined;
                this.startInitialLoad(filename, dir);
            }
        }

        if (this.props.onVideoStateChange &&
            (prevState.currentTime !== this.state.currentTime ||
             prevState.duration !== this.state.duration ||
             prevState.isPlaying !== this.state.isPlaying ||
             prevState.frameRate !== this.state.frameRate)) {
            this.props.onVideoStateChange({
                currentTime: this.state.currentTime,
                duration: this.state.duration,
                isPlaying: this.state.isPlaying,
                frameRate: this.state.frameRate
            });
        }

        if (
            typeof this.props.jumpToFrameIndex === 'number' &&
            this.props.jumpToFrameIndex !== prevProps.jumpToFrameIndex &&
            this.playerRef &&
            this.state.videoSourceLoaded
        ) {
            const targetFrame = this.props.jumpToFrameIndex;
            const time = targetFrame / this.state.frameRate;
            const cached = this.frameCache?.get(targetFrame) ?? null;
            if (cached) {
                this.paintOverlayFrame(cached, targetFrame);
            }
            this.internalSeekInProgress = true;
            this.overlayPaintedFromCacheForCurrentSeek = !!cached;
            this.playerRef.currentTime = time;
            this.setClock(time);
            this.setState({ currentTime: time, isPlaying: false });
            if (this.props.jumpToFrame) {
                this.props.jumpToFrame(null);
            }
        }

        // Debounced auto-save when labels change
        if (prevProps.imageData.labelRects !== currentImageData.labelRects) {
            this.debouncedSaveLabels();
        }
    }

    componentWillUnmount() {
        if (this.updateImageDataTimeout) {
            clearTimeout(this.updateImageDataTimeout);
        }
        if (this.saveTimeout) {
            clearTimeout(this.saveTimeout);
            this.saveLabelsToBackend(); // Final save
        }
        if (this.frameCache) {
            this.frameCache.clear();
            this.frameCache = null;
        }
        if (this.playerRef) {
            this.playerRef.destroy();
            this.playerRef = null;
        }
        document.removeEventListener('keydown', this.handleKeyDown);
    }

    private paintOverlayFrame = (bitmap: ImageBitmap, frameIndex: number) => {
        const canvas = this.overlayCanvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
            canvas.width = bitmap.width;
            canvas.height = bitmap.height;
        }
        ctx.drawImage(bitmap, 0, 0);
        canvas.style.display = 'block';
        this.overlayDisplayedFrame = frameIndex;
    };

    // Synchronously copy whatever the <video> element is currently showing
    // onto the overlay canvas. Used on pause and after external seeks so the
    // overlay always has up-to-date content covering the real video; that way
    // a subsequent step is a pure canvas content swap (no display:none↔block
    // toggle, no compositor layer flash).
    private captureCurrentFrameToOverlay = (): boolean => {
        const videoEl = this.videoRef.current;
        const canvas = this.overlayCanvasRef.current;
        if (!videoEl || !canvas) return false;
        const ctx = canvas.getContext('2d');
        if (!ctx) return false;
        const w = videoEl.videoWidth;
        const h = videoEl.videoHeight;
        if (w === 0 || h === 0) return false;
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
        try {
            ctx.drawImage(videoEl, 0, 0, w, h);
        } catch {
            return false;
        }
        canvas.style.display = 'block';
        if (this.playerRef) {
            // Floor (not round): frame N is shown for time [N/fps, (N+1)/fps),
            // matching stepFrame / jumpToTime. Mixing floor and round here
            // makes the overlay-vs-real frame check in handleRealVideoSeeked
            // false-negative when the playhead is past mid-frame.
            this.overlayDisplayedFrame = Math.floor(
                this.playerRef.currentTime * this.state.frameRate
            );
        }
        return true;
    };

    private hideOverlay = () => {
        const canvas = this.overlayCanvasRef.current;
        if (canvas) canvas.style.display = 'none';
        this.overlayDisplayedFrame = null;
    };

    private isVideoPaused = (): boolean => {
        // Use the underlying <video>.paused for ground truth — state.isPlaying
        // is event-driven and may briefly lag during seek-driven transitions.
        return this.videoRef.current?.paused ?? true;
    };

    private resetPlaybackSpeed = () => {
        if (this.playerRef) {
            try {
                this.playerRef.speed = 1;
            } catch {}
        }
        const videoEl = this.videoRef.current;
        if (videoEl && videoEl.playbackRate !== 1) {
            videoEl.playbackRate = 1;
        }
    };

    private handleRealVideoSeeking = () => {
        // If we initiated this seek (step / jump), keep the overlay on top so
        // there is no visible flicker. Otherwise the seek is external (most
        // commonly the user dragging Plyr's progress bar) and they expect to
        // see the live video update — drop the overlay so it doesn't freeze.
        if (this.internalSeekInProgress) return;
        if (!this.isVideoPaused()) return;
        this.hideOverlay();
    };

    private handleRealVideoSeeked = () => {
        if (!this.playerRef) return;
        const wasInternal = this.internalSeekInProgress;
        const wasCacheHit = this.overlayPaintedFromCacheForCurrentSeek;
        this.internalSeekInProgress = false;
        this.overlayPaintedFromCacheForCurrentSeek = false;

        // Live playback: hide the overlay so the real video shows through.
        if (!this.isVideoPaused()) {
            this.hideOverlay();
            return;
        }

        // Internal seek that painted from the cache: overlay already shows
        // the right frame. Don't recapture from the main <video> — even when
        // the logical frame matches, the cache and main video decode
        // independently, so the redraw swaps pixels (subtle color/scaling
        // differences) and the eye reads it as a same-frame flicker.
        if (wasInternal && wasCacheHit) return;

        // Cache miss (overlay still shows the previous frame), or external
        // drag-seek (overlay was hidden in `seeking`): refresh from the real
        // video so the overlay catches up to the new position.
        this.captureCurrentFrameToOverlay();
    };

    stepFrame = (direction: 'forward' | 'backward') => {
        if (this.playerRef && this.state.videoSourceLoaded) {
            const currentTime = this.playerRef.currentTime;
            const pts = this.frameTimestamps;
            // PTS path: step by exactly one decoded frame index — correct
            // for VFR sources where neighboring frames aren't 1/fps apart.
            // Fallback path mirrors the legacy `±1/fps` math for tests and
            // for sources whose PTS index didn't load.
            let newTime: number;
            let targetFrame: number;
            if (pts && pts.length > 0) {
                // The browser's native demuxer and decord can resolve the
                // container's time base to slightly different floats, so
                // after seeking forward to ``pts[i]`` the player can report
                // ``currentTime`` a hair *below* ``pts[i]``. Without the
                // snap, ``floorPtsIndex`` then returns ``i-1`` and forward-
                // step targets ``i`` again — the same frame we're already
                // showing — so the right arrow looks frozen while the left
                // arrow keeps working. 0.5 ms is well above any FP/time-
                // base rounding (~ns) and well below any real frame
                // interval (≥ 4 ms at 240 fps).
                let cur = floorPtsIndex(pts, currentTime);
                if (cur + 1 < pts.length && pts[cur + 1] - currentTime < 5e-4) {
                    cur += 1;
                }
                const delta = direction === 'forward' ? 1 : -1;
                targetFrame = Math.min(Math.max(cur + delta, 0), pts.length - 1);
                newTime = pts[targetFrame];
            } else {
                const frameTime = 1 / this.state.frameRate;
                newTime = direction === 'forward'
                    ? Math.min(currentTime + frameTime, this.playerRef.duration)
                    : Math.max(currentTime - frameTime, 0);
                targetFrame = Math.floor(newTime * this.state.frameRate);
            }

            // Paint the overlay BEFORE assigning currentTime. The canvas
            // covers the real <video> while it's still decoding the new
            // frame, so the user sees an instant content swap on the same
            // compositing layer instead of a hide→reveal flicker.
            let cached: ImageBitmap | null = null;
            if (this.frameCache) {
                cached = this.frameCache.get(targetFrame);
                if (cached) {
                    this.paintOverlayFrame(cached, targetFrame);
                }
            }

            // Mark this seek as ours BEFORE the assignment, so the seeking
            // handler doesn't tear down the overlay we just painted. Also
            // record whether the overlay holds a cache bitmap, so the seeked
            // handler can keep it instead of recapturing from the main video.
            this.internalSeekInProgress = true;
            this.overlayPaintedFromCacheForCurrentSeek = !!cached;
            this.playerRef.currentTime = newTime;
            this.setClock(newTime);

            if (!cached && this.frameCache) {
                // Out of cached window — recenter prefetch so the next step
                // has a chance to hit. The overlay still shows the previous
                // frame for now; handleRealVideoSeeked refreshes it from the
                // real video once the seek lands.
                this.frameCache.prefetch(targetFrame);
            }

            this.setState({ currentTime: newTime, isPlaying: false }, () => {
                if (Math.abs(this.props.imageData.frameIndex - targetFrame) > 0) {
                    this.debouncedUpdateImageData({
                        ...this.props.imageData,
                        timestamp: newTime,
                        frameIndex: targetFrame
                    });
                }
            });
        }
    };

    stepSeconds = (direction: 'forward' | 'backward') => {
        if (this.playerRef && this.state.videoSourceLoaded) {
            const currentTime = this.playerRef.currentTime;
            const newTime = direction === 'forward'
                ? Math.min(currentTime + 10, this.playerRef.duration)
                : Math.max(currentTime - 10, 0);
            this.playerRef.currentTime = newTime;
            this.setClock(newTime);
            this.setState({ currentTime: newTime });
        }
    };

    jumpBoundary = (direction: 'forward' | 'backward') => {
        if (!this.playerRef || !this.state.videoSourceLoaded) return;
        const currentTime = this.playerRef.currentTime;
        const eps = 0.5 / (this.state.frameRate || 30);
        const events = this.collectBoundaryEvents();
        const candidates = direction === 'forward'
            ? events.filter(e => e.time > currentTime + eps)
            : events.filter(e => e.time < currentTime - eps);
        if (!candidates.length) return;
        const target = direction === 'forward'
            ? candidates.reduce((a, b) => (b.time < a.time ? b : a))
            : candidates.reduce((a, b) => (b.time > a.time ? b : a));
        const fmt = TimeUtil.formatTimeWithFrame(target.time, this.state.frameRate).formattedTime;
        this.jumpToTime(fmt);
    };

    togglePlay = () => {
        if (this.playerRef) {
            this.playerRef.togglePlay();
        }
    };

    handlePlyrTimeUpdate = (currentTime: number) => {
        // Imperative path: push into the shared PlayheadClock on every tick
        // (no React re-render). The rAF loop inside <Playhead> reads it.
        this.setClock(currentTime);
        // Coarse React path: setState is throttled to 250 ms. Used by
        // useAutoScroll and by the ongoing-clip width calculation in
        // CustomTimeline — both tolerate 4 Hz cadence.
        this.throttledUpdate(() => {
            this.setState({ currentTime });
        }, 250);
    };

    handlePlyrPlay = () => {
        this.setState({ isPlaying: true });
        // Playback invalidates the frame cache — memory is only held while paused.
        if (this.frameCache) {
            this.frameCache.clear();
        }
        this.internalSeekInProgress = false;
        this.overlayPaintedFromCacheForCurrentSeek = false;
        this.hideOverlay();
    };

    handlePlyrPause = () => {
        this.setState({ isPlaying: false });
        this.internalSeekInProgress = false;
        this.overlayPaintedFromCacheForCurrentSeek = false;
        // Cover the real video with the overlay immediately. The next step
        // (or any other paused-state update) will swap canvas content on the
        // same compositing layer rather than re-mounting it from scratch.
        this.captureCurrentFrameToOverlay();
        if (this.frameCache && this.playerRef) {
            // Floor matches stepFrame's convention so the cache window is
            // centered on the frame the user is actually looking at.
            const centerFrame = Math.floor(this.playerRef.currentTime * this.state.frameRate);
            this.frameCache.prefetch(centerFrame);
        }
    };

    handlePlyrDurationChange = (duration: number) => {
        // Metadata is in (duration known, seeking allowed). The video-stream
        // load stage stays 'active' until `canplay` so the staged overlay
        // keeps covering the fill-initial-buffer gap.
        this.setState({
            duration,
            videoSourceLoaded: true,
        });
    };

    jumpToTime = (timestamp: string) => {
        if (this.playerRef && this.state.videoSourceLoaded) {
            const seconds = TimeUtil.parseTimestamp(timestamp);
            // Floor matches stepFrame: frame N is shown for [N/fps, (N+1)/fps),
            // and the FrameCache is keyed by floor-derived frame indices. Using
            // round here would make jump targets miss the cache by one when the
            // timestamp's fractional frame is past the half-frame mark.
            const targetFrame = Math.floor(seconds * this.state.frameRate);
            // Paint a cached bitmap onto the overlay BEFORE seeking so the
            // jump is visually instant. On a cache miss, the overlay keeps
            // showing the previous frame; handleRealVideoSeeked refreshes
            // it from the real video once the seek completes.
            const cached = this.frameCache?.get(targetFrame) ?? null;
            if (cached) {
                this.paintOverlayFrame(cached, targetFrame);
            }
            this.internalSeekInProgress = true;
            this.overlayPaintedFromCacheForCurrentSeek = !!cached;
            this.playerRef.currentTime = seconds;
            this.setClock(seconds);
            if (this.frameCache && !this.state.isPlaying) {
                this.frameCache.prefetch(targetFrame);
            }
            this.setState({
                currentTime: seconds,
                isPlaying: false
            });
        }
    };

    retryLoad = () => {
        this.setState({ hasError: false, isBuffering: true });
        if (this.props.imageData.videoUrl) {
            this.updateVideoSource(this.props.imageData.videoUrl);
        }
    };

    private collectBoundaryEvents = (): { id: string; time: number; type: 'start' | 'end' }[] => {
        const { imageData, focusedLabelNameId } = this.props;
        const fr = this.state.frameRate || 30;
        // Focus mode narrows boundary navigation to the picked behavior so
        // Shift+Arrow / Cmd+Arrow only step between *its* clip edges. Without
        // this filter the user would hop into hidden clips of other behaviors.
        const rects = focusedLabelNameId
            ? (imageData.labelRects || []).filter(r => r.labelId === focusedLabelNameId)
            : (imageData.labelRects || []);
        return rects.flatMap(rect => {
            const out: { id: string; time: number; type: 'start' | 'end' }[] = [];
            let startTime: number | null = null;
            if (rect.timestamp) {
                startTime = TimeUtil.parseTimestamp(rect.timestamp);
            } else if (typeof rect.frame === 'number') {
                startTime = rect.frame / fr;
            }
            if (startTime != null && Number.isFinite(startTime)) {
                out.push({ id: rect.id, time: startTime, type: 'start' });
            }
            let endTime: number | null = null;
            if (rect.endTimestamp) {
                endTime = TimeUtil.parseTimestamp(rect.endTimestamp);
            } else if (typeof rect.endFrame === 'number') {
                endTime = rect.endFrame / fr;
            }
            if (endTime != null && Number.isFinite(endTime)) {
                out.push({ id: rect.id, time: endTime, type: 'end' });
            }
            return out;
        });
    };

    handleKeyDown = (event: KeyboardEvent) => {
        if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;

        const { code } = event;
        const { labelNames, imageData } = this.props;
        // Read the live player time. `this.state.currentTime` is throttled to
        // 250 ms by `handlePlyrTimeUpdate`, so during/just-after playback it
        // can lag reality by up to a frame on a 30 fps timeline — enough to
        // place the cursor *inside* a short clip while the state still says
        // it's before it, which made Shift+Right snap to the clip's *start*
        // (already behind us) or appear to do nothing.
        const currentTime = this.playerRef && this.state.videoSourceLoaded
            ? this.playerRef.currentTime
            : this.state.currentTime;
        // Half-frame epsilon: treat boundaries within half a frame of the
        // playhead as "already passed" so Shift+Right advances instead of
        // re-selecting the boundary we're sitting on. Matters for short
        // behaviors whose start/end land within one frame of each other.
        const eps = 0.5 / (this.state.frameRate || 30);

        if (code === 'ArrowLeft' || code === 'ArrowRight') {
            if (event.shiftKey) {
                event.preventDefault();
                this.jumpBoundary(code === 'ArrowLeft' ? 'backward' : 'forward');
                return;
            }

            if (event.metaKey || event.ctrlKey) {
                event.preventDefault();
                const events = this.collectBoundaryEvents();
                const futureEvents = events.filter(e => e.time > currentTime + eps);
                const pastEvents = events.filter(e => e.time < currentTime - eps);
                let target;
                if (code === 'ArrowLeft') {
                    if (futureEvents.length === 0) return;
                    target = futureEvents.reduce((a, b) => (b.time < a.time ? b : a));
                } else if (code === 'ArrowRight') {
                    if (pastEvents.length === 0) return;
                    target = pastEvents.reduce((a, b) => (b.time > a.time ? b : a));
                } else {
                    return;
                }
                const snapped = snapTime(currentTime, this.frameTimestamps, this.state.frameRate);
                const updatedRects = imageData.labelRects.map(rect => {
                    if (rect.id === target.id) {
                        const fmt = TimeUtil.formatTimeWithFrame(snapped.time, this.state.frameRate).formattedTime;
                        return target.type === 'start'
                            ? { ...rect, timestamp: fmt, frame: snapped.frame }
                            : { ...rect, endTimestamp: fmt, endFrame: snapped.frame };
                    }
                    return rect;
                });
                this.props.updateImageDataById(imageData.id, { ...imageData, labelRects: updatedRects });
                return;
            }

            const direction: 'forward' | 'backward' = code === 'ArrowRight' ? 'forward' : 'backward';
            this.stepFrame(direction);
            event.preventDefault();
            return;
        }

        switch (code) {
            case 'Space':
                event.preventDefault();
                this.togglePlay();
                break;
            case 'Escape':
                event.preventDefault();
                this.timeChangeId = null;
                break;
            default:
                // Number keys 1-9 switch the active subject. Behavior shortcuts
                // are filtered to [a-z] in InsertLabelNamesPopup so digits
                // never collide with a behavior key, but we still match
                // explicitly to keep that contract local to this handler.
                if (!event.metaKey && !event.ctrlKey && !event.altKey
                    && /^[1-9]$/.test(event.key)) {
                    const idx = parseInt(event.key, 10) - 1;
                    const target = this.props.subjects[idx];
                    if (target) {
                        event.preventDefault();
                        this.props.updateActiveSubjectId(target.id);
                    }
                    break;
                }

                const matchingLabel = labelNames.find(label =>
                    label.shortcut && label.shortcut.toLowerCase() === event.key.toLowerCase()
                );

                if (matchingLabel) {
                    event.preventDefault();
                    const updatedData = toggleBehaviorClip(
                        matchingLabel,
                        imageData,
                        currentTime,
                        this.state.frameRate,
                        this.props.activeSubjectId,
                        this.frameTimestamps,
                    );
                    this.props.updateImageDataById(imageData.id, updatedData);
                }
                break;
        }
    };

    initializePlayer = () => {
        if (this.playerRef) return;
        // React 18 StrictMode (dev) simulates an unmount/remount on first mount.
        // Plyr.destroy() in componentWillUnmount swaps the wrapped <video> out
        // for a clone of the original, leaving this.videoRef.current pointing
        // to a detached node whose stale `.plyr` flag would make a second
        // `new Plyr(...)` bail at the "Target already setup" guard. Always work
        // off the live DOM node and clear any leftover `.plyr` before init.
        const liveVideo = this.videoRef.current?.isConnected
            ? this.videoRef.current
            : document.querySelector<HTMLVideoElement>('.plyr-video-container .plyr-video');
        if (!liveVideo) return;
        if ((liveVideo as any).plyr) {
            (liveVideo as any).plyr = null;
        }

        this.setState({ isBuffering: true });

        this.playerRef = new Plyr(liveVideo, {
            controls: [
                'play-large',
                'play',
                'progress',
                'current-time',
                'duration',
                'mute',
                'volume',
                'settings',
                'fullscreen'
            ],
            settings: ['speed'],
            speed: {
                selected: 1,
                options: [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 3, 4, 5]
            },
            storage: {
                enabled: false
            },
            keyboard: {
                focused: false,
                global: false
            },
            tooltips: {
                controls: true,
                seek: true
            },
            displayDuration: true,
            invertTime: false,
            toggleInvert: false,
            ratio: null,
            clickToPlay: true,
            hideControls: false,
            resetOnEnd: false,
            disableContextMenu: false
        });

        const videoUrl = this.props.imageData.videoUrl;
        this.playerRef.source = {
            type: 'video',
            sources: [{ src: videoUrl, type: this.getVideoType(videoUrl) }]
        };
        this.resetPlaybackSpeed();
        const cacheVideo = this.cacheVideoRef.current;
        if (cacheVideo && videoUrl) {
            cacheVideo.src = videoUrl;
            cacheVideo.load();
        }

        const player = this.playerRef;

        player.on('timeupdate', () => {
            this.handlePlyrTimeUpdate(player.currentTime);
        });

        player.on('loadedmetadata', () => {
            this.handlePlyrDurationChange(player.duration);
            // Pull the file's actual fps from props (set by FileBrowser
            // off the backend's /api/files response). HTMLVideoElement
            // doesn't expose fps, so this is the canonical source.
            const fr =
                (this.props.imageData?.frameRate && this.props.imageData.frameRate > 0)
                    ? this.props.imageData.frameRate
                    : 30;
            this.resetPlaybackSpeed();
            this.setState({ frameRate: fr, videoSourceLoaded: true, isBuffering: false });
            // (Re)build the frame cache whenever metadata loads — this fires
            // both on first source and on source change. The ±N-frame
            // window is held constant; the second arg is the fps used to
            // map seek-time to frame index in the cache, so it must match
            // the file's real fps.
            if (this.frameCache) {
                this.frameCache.clear();
            }
            const cacheVideoEl = this.cacheVideoRef.current;
            if (cacheVideoEl) {
                this.frameCache = new FrameCache(cacheVideoEl, fr, {
                    radius: 30,
                    getRenderSize: () => {
                        const videoEl = this.videoRef.current;
                        if (!videoEl) return null;
                        return {
                            width: videoEl.clientWidth,
                            height: videoEl.clientHeight,
                        };
                    },
                });
            }
        });

        player.on('play', () => this.handlePlyrPlay());
        player.on('pause', () => this.handlePlyrPause());
        player.on('seeking', this.handleRealVideoSeeking);
        player.on('seeked', this.handleRealVideoSeeked);

        player.on('waiting', () => {
            this.setState({ isBuffering: true });
        });

        player.on('canplay', () => {
            this.setState(prev => ({
                isBuffering: false,
                loadStages: { ...prev.loadStages, videoStream: 'done' },
            }));
        });

        player.on('playing', () => {
            this.setState(prev => ({
                isBuffering: false,
                loadStages: { ...prev.loadStages, videoStream: 'done' },
            }));
        });

        player.on('error', (event) => {
            this.setState({
                hasError: true,
                isBuffering: false,
                errorMessage: 'Failed to load video. Check the server connection.'
            });
        });
    };

    updateVideoSource = (videoUrl: string) => {
        if (this.playerRef && videoUrl) {
            this.setState({ isBuffering: true, hasError: false });
            this.playerRef.source = {
                type: 'video',
                sources: [{ src: videoUrl, type: this.getVideoType(videoUrl) }]
            };
            this.resetPlaybackSpeed();
            // Keep the hidden cache video in sync so prefetch sees the same asset.
            // Browsers typically share HTTP cache for same-URL requests, so this
            // is one range-request set rather than doubled network cost.
            const updatedCacheVideo = this.cacheVideoRef.current;
            if (updatedCacheVideo && updatedCacheVideo.src !== videoUrl) {
                updatedCacheVideo.src = videoUrl;
                updatedCacheVideo.load();
            }
            if (this.frameCache) {
                this.frameCache.clear();
            }
            this.overlayPaintedFromCacheForCurrentSeek = false;
            this.hideOverlay();
        }
    };

    getVideoType = (url: string): string => {
        if (!url) return 'video/mp4';
        const extension = url.split('.').pop()?.split('?')[0]?.toLowerCase();
        switch (extension) {
            case 'mp4': return 'video/mp4';
            case 'webm': return 'video/webm';
            case 'ogg': return 'video/ogg';
            case 'mov': return 'video/quicktime';
            case 'avi': return 'video/x-msvideo';
            default: return 'video/mp4';
        }
    };

    private renderStageIcon(status: LoadStageStatus) {
        if (status === 'done') return <span className='LoadStageIcon LoadStageIcon--done' aria-label='Done'>✓</span>;
        if (status === 'fallback') return <span className='LoadStageIcon LoadStageIcon--warn' aria-label='Unavailable'>!</span>;
        if (status === 'active') return <span className='LoadStageSpinner' aria-label='Loading' />;
        return <span className='LoadStageIcon LoadStageIcon--pending' aria-hidden='true'>·</span>;
    }

    render() {
        const { isBuffering, hasError, errorMessage, loadStages } = this.state;
        // Initial load = first time we're showing this video. Stage list
        // overlay covers it. We keep the simple buffering spinner for the
        // in-playback `waiting`/`canplay` cycle, since that's a transient
        // network event the user already understands.
        const initialLoading =
            loadStages.videoStream !== 'done'
            || (loadStages.frameIndex !== 'done' && loadStages.frameIndex !== 'fallback');
        const frameIndexLabel = loadStages.frameIndex === 'fallback'
            ? 'Frame index unavailable — using nominal fps'
            : 'Indexing frames';

        return (
            <div className="Editor" style={{ width: '100%', height: '100%' }}>
                <div className="VideoContainer" style={{ width: '100%', height: '100%', position: 'relative' }}>
                    {initialLoading && !hasError && (
                        <div className='BufferingOverlay LoadingOverlay'>
                            <ul className='LoadStageList'>
                                <li className={`LoadStage LoadStage--${loadStages.videoStream}`}>
                                    {this.renderStageIcon(loadStages.videoStream)}
                                    <span className='LoadStageLabel'>Video stream</span>
                                </li>
                                <li className={`LoadStage LoadStage--${loadStages.frameIndex}`}>
                                    {this.renderStageIcon(loadStages.frameIndex)}
                                    <span className='LoadStageLabel'>{frameIndexLabel}</span>
                                </li>
                            </ul>
                        </div>
                    )}
                    {!initialLoading && isBuffering && !hasError && (
                        <div className="BufferingOverlay">
                            <div className="BufferingSpinner" />
                            <span>Buffering…</span>
                        </div>
                    )}
                    {hasError && (
                        <div className="ErrorOverlay">
                            <span className="ErrorMessage">{errorMessage}</span>
                            <button type='button' className="RetryButton" onClick={this.retryLoad}>Retry</button>
                        </div>
                    )}
                    <div className="plyr-video-container" style={{ width: '100%', height: '100%' }}>
                        <video
                            ref={this.videoRef}
                            className="plyr-video"
                            playsInline
                            preload="metadata"
                            style={{ width: '100%', height: '100%' }}
                        >
                            <p>Your browser doesn't support HTML5 video.</p>
                        </video>
                        <video
                            ref={this.cacheVideoRef}
                            className="cache-video"
                            muted
                            playsInline
                            preload="metadata"
                            aria-hidden="true"
                        />
                        <canvas
                            ref={this.overlayCanvasRef}
                            className="frame-overlay"
                        />
                    </div>
                </div>
            </div>
        );
    }
}

const mapDispatchToProps = {
    updateImageDataById,
    jumpToFrame,
    updateActiveSubjectId,
    markVideoHasCsv,
};

const mapStateToProps = (state: AppState) => ({
    labelNames: state.labels.labels,
    subjects: state.labels.subjects,
    activeSubjectId: state.labels.activeSubjectId,
    focusedLabelNameId: state.labels.focusedLabelNameId,
    jumpToFrameIndex: state.general.jumpToFrameIndex,
});

const ConnectedEditor = connect(
    mapStateToProps,
    mapDispatchToProps,
    null,
    { forwardRef: true }
)(Editor);

const ForwardedEditor = forwardRef<any, any>((props, ref) => <ConnectedEditor {...props} ref={ref} />);

export default ForwardedEditor;
