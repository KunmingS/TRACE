import React, { createRef, forwardRef } from 'react';
import './Editor.scss';
import {ImageData, LabelName} from '../../../store/labels/types';
import {AppState} from '../../../store';
import {connect} from 'react-redux';
import {updateImageDataById} from '../../../store/labels/actionCreators';
import {ISize} from '../../../interfaces/ISize';
import { TimeUtil } from '../../../utils/TimeUtil';
import { toggleBehaviorClip } from '../../../utils/BehaviorUtil';
import { jumpToFrame } from '../../../store/general/actionCreators';
import { API_URL } from '../../../config';
import Plyr from 'plyr';
import 'plyr/dist/plyr.css';

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
    jumpToFrameIndex?: number | null;
    jumpToFrame?: (frameIndex: number | null) => void;
}

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
}

class Editor extends React.Component<IProps, IState> {
    videoRef = createRef<HTMLVideoElement>();
    playerRef: Plyr | null = null;
    timeChangeId: string | null = null;
    private updateImageDataTimeout: number | null = null;
    private lastUpdateTime: number = 0;
    private saveTimeout: number | null = null;
    private lastSavedRectsJson: string = '';

    constructor(props) {
        super(props);
        this.state = {
            duration: 0,
            currentTime: 0,
            isPlaying: false,
            videoUrl: null,
            frameRate: 30,
            videoSourceLoaded: false,
            isBuffering: false,
            hasError: false,
            errorMessage: '',
        };
    }

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

        const rectsJson = JSON.stringify(rectsToSave.map(r => ({
            behavior: r.behavior,
            timestamp: TimeUtil.parseTimestamp(r.timestamp!),
            endTimestamp: TimeUtil.parseTimestamp(r.endTimestamp!)
        })));

        // Skip if nothing changed
        if (rectsJson === this.lastSavedRectsJson) return;

        const dir = (this.props.imageData as any).videoPath;
        if (!dir) return;
        const filename = this.props.imageData.fileData.name;

        fetch(
            `${API_URL}/api/files/${filename}/labels?dir=${encodeURIComponent(dir)}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    videoPath: dir,
                    labelRects: rectsToSave.map(r => ({
                        behavior: r.behavior,
                        timestamp: TimeUtil.parseTimestamp(r.timestamp!),
                        endTimestamp: TimeUtil.parseTimestamp(r.endTimestamp!)
                    }))
                })
            }
        )
        .then(res => {
            if (res.ok) {
                this.lastSavedRectsJson = rectsJson;
            }
        })
        .catch(() => {});
    };

    componentDidMount() {
        if (this.props.imageData.videoUrl) {
            this.setState({ videoUrl: this.props.imageData.videoUrl });
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
            const time = this.props.jumpToFrameIndex / this.state.frameRate;
            this.playerRef.currentTime = time;
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
        if (this.playerRef) {
            this.playerRef.destroy();
            this.playerRef = null;
        }
        document.removeEventListener('keydown', this.handleKeyDown);
    }

    stepFrame = (direction: 'forward' | 'backward') => {
        if (this.playerRef && this.state.videoSourceLoaded) {
            const frameTime = 1 / this.state.frameRate;
            const currentTime = this.playerRef.currentTime;
            const newTime = direction === 'forward'
                ? Math.min(currentTime + frameTime, this.playerRef.duration)
                : Math.max(currentTime - frameTime, 0);
            this.playerRef.currentTime = newTime;

            this.setState({ currentTime: newTime, isPlaying: false }, () => {
                const frameIndex = Math.floor(newTime * this.state.frameRate);
                if (Math.abs(this.props.imageData.frameIndex - frameIndex) > 0) {
                    this.debouncedUpdateImageData({
                        ...this.props.imageData,
                        timestamp: newTime,
                        frameIndex: frameIndex
                    });
                }
            });
        }
    };

    stepSeconds = (direction: 'forward' | 'backward') => {
        if (this.playerRef && this.state.videoSourceLoaded) {
            const currentTime = this.playerRef.currentTime;
            const newTime = direction === 'forward'
                ? Math.min(currentTime + 3, this.playerRef.duration)
                : Math.max(currentTime - 3, 0);
            this.playerRef.currentTime = newTime;
            this.setState({ currentTime: newTime });
        }
    };

    togglePlay = () => {
        if (this.playerRef) {
            this.playerRef.togglePlay();
        }
    };

    handlePlyrTimeUpdate = (currentTime: number) => {
        this.throttledUpdate(() => {
            this.setState({ currentTime });
        }, 100);
    };

    handlePlyrPlay = () => {
        this.setState({ isPlaying: true });
    };

    handlePlyrPause = () => {
        this.setState({ isPlaying: false });
    };

    handlePlyrDurationChange = (duration: number) => {
        this.setState({
            duration,
            videoSourceLoaded: true,
            isBuffering: false,
        });
    };

    jumpToTime = (timestamp: string) => {
        if (this.playerRef && this.state.videoSourceLoaded) {
            const seconds = TimeUtil.parseTimestamp(timestamp);
            this.playerRef.currentTime = seconds;
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

    handleKeyDown = (event: KeyboardEvent) => {
        if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;

        const { code } = event;
        const { labelNames, imageData } = this.props;
        const currentTime = this.state.currentTime;

        if (code === 'ArrowLeft' || code === 'ArrowRight') {
            if (event.shiftKey) {
                event.preventDefault();
                const events = (imageData.labelRects || []).flatMap(rect => [
                    { id: rect.id, time: TimeUtil.parseTimestamp(rect.timestamp!), type: 'start' },
                    ...(rect.endTimestamp != null ? [{ id: rect.id, time: TimeUtil.parseTimestamp(rect.endTimestamp!), type: 'end' }] : [])
                ]);
                if (code === 'ArrowLeft') {
                    const prev = events.filter(e => e.time < currentTime);
                    if (prev.length) {
                        const nearest = prev.reduce((a, b) => (b.time > a.time ? b : a));
                        const fmt = TimeUtil.formatTimeWithFrame(nearest.time, this.state.frameRate).formattedTime;
                        this.jumpToTime(fmt);
                    }
                } else {
                    const next = events.filter(e => e.time > currentTime);
                    if (next.length) {
                        const nearest = next.reduce((a, b) => (b.time < a.time ? b : a));
                        const fmt = TimeUtil.formatTimeWithFrame(nearest.time, this.state.frameRate).formattedTime;
                        this.jumpToTime(fmt);
                    }
                }
                return;
            }

            if (event.metaKey || event.ctrlKey) {
                event.preventDefault();
                const events = (imageData.labelRects || []).flatMap(rect => [
                    { id: rect.id, time: TimeUtil.parseTimestamp(rect.timestamp!), type: 'start' },
                    ...(rect.endTimestamp != null ? [{ id: rect.id, time: TimeUtil.parseTimestamp(rect.endTimestamp!), type: 'end' }] : [])
                ]);
                const futureEvents = events.filter(e => e.time > currentTime);
                const pastEvents = events.filter(e => e.time < currentTime);
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
                const updatedRects = imageData.labelRects.map(rect => {
                    if (rect.id === target.id) {
                        const fmt = TimeUtil.formatTimeWithFrame(currentTime, this.state.frameRate).formattedTime;
                        return target.type === 'start'
                            ? { ...rect, timestamp: fmt }
                            : { ...rect, endTimestamp: fmt };
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
                const matchingLabel = labelNames.find(label =>
                    label.shortcut && label.shortcut.toLowerCase() === event.key.toLowerCase()
                );

                if (matchingLabel) {
                    event.preventDefault();
                    const updatedData = toggleBehaviorClip(matchingLabel, imageData, currentTime, this.state.frameRate);
                    this.props.updateImageDataById(imageData.id, updatedData);
                }
                break;
        }
    };

    initializePlayer = () => {
        if (this.videoRef.current && !this.playerRef) {
            this.setState({ isBuffering: true });

            this.playerRef = new Plyr('video', {
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

            const player = this.playerRef;

            player.on('timeupdate', () => {
                this.handlePlyrTimeUpdate(player.currentTime);
            });

            player.on('loadedmetadata', () => {
                this.handlePlyrDurationChange(player.duration);
                this.setState({ frameRate: 30, videoSourceLoaded: true, isBuffering: false });
            });

            player.on('play', () => this.handlePlyrPlay());
            player.on('pause', () => this.handlePlyrPause());

            player.on('waiting', () => {
                this.setState({ isBuffering: true });
            });

            player.on('canplay', () => {
                this.setState({ isBuffering: false });
            });

            player.on('playing', () => {
                this.setState({ isBuffering: false });
            });

            player.on('error', (event) => {
                this.setState({
                    hasError: true,
                    isBuffering: false,
                    errorMessage: 'Failed to load video. Check the server connection.'
                });
            });
        }
    };

    updateVideoSource = (videoUrl: string) => {
        if (this.playerRef && videoUrl) {
            this.setState({ isBuffering: true, hasError: false });
            this.playerRef.source = {
                type: 'video',
                sources: [{ src: videoUrl, type: this.getVideoType(videoUrl) }]
            };
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

    render() {
        const { isBuffering, hasError, errorMessage } = this.state;

        return (
            <div className="Editor" style={{ width: '100%', height: '100%' }}>
                <div className="VideoContainer" style={{ width: '100%', height: '100%', position: 'relative' }}>
                    {isBuffering && !hasError && (
                        <div className="BufferingOverlay">
                            <div className="BufferingSpinner" />
                            <span>Loading video...</span>
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
                    </div>
                </div>
            </div>
        );
    }
}

const mapDispatchToProps = {
    updateImageDataById,
    jumpToFrame,
};

const mapStateToProps = (state: AppState) => ({
    labelNames: state.labels.labels,
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
