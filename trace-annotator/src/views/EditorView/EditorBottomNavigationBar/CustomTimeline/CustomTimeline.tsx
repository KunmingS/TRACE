import React, {useCallback, useRef, useState, useEffect} from 'react';
import './CustomTimeline.scss';
import {TimelineTrack} from './types';
import {useAutoScroll} from './useAutoScroll';
import classNames from 'classnames';
import TimeRuler from '../TimeRuler/TimeRuler';
import Playhead from '../Playhead/Playhead';
import {TimeUtil} from '../../../../utils/TimeUtil';
import MiniMap from '../MiniMap/MiniMap';
import ClipHandle from '../ClipHandle/ClipHandle';
import {PlayheadClock} from '../playheadClock';

interface CustomTimelineProps {
    tracks: TimelineTrack[];
    currentTime: number;
    duration: number;
    scaleWidth: number;
    onJumpToTime: (time: string) => void;
    isCompactLayout: boolean;
    activeLabelId: string | null;
    onSelectClip: (clipId: string | null) => void;
    playheadClock?: PlayheadClock;
    rowLabelsRef?: React.RefObject<HTMLDivElement>;
}

const START_LEFT = 20;

const CustomTimeline: React.FC<CustomTimelineProps> = ({
    tracks,
    currentTime,
    duration,
    scaleWidth,
    onJumpToTime,
    isCompactLayout,
    activeLabelId,
    onSelectClip,
    playheadClock,
    rowLabelsRef
}) => {
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const scrollRafRef = useRef<number | null>(null);
    const [scrollLeft, setScrollLeft] = useState(0);
    const [containerWidth, setContainerWidth] = useState(0);

    const totalWidth = START_LEFT + duration * scaleWidth;
    const viewLeft = scrollLeft - 200;
    const viewRight = scrollLeft + containerWidth + 200;

    const secondsToPixels = (t: number): number => START_LEFT + t * scaleWidth;

    useAutoScroll(currentTime, duration, scaleWidth, scrollContainerRef);

    useEffect(() => {
        const container = scrollContainerRef.current;
        if (!container) return undefined;

        const handleScroll = () => {
            // Mirror vertical scroll into the sidebar's label column so the
            // label list and the track list stay row-aligned when the behavior
            // count exceeds the visible area. Guarded against feedback loops
            // by only writing when the values actually differ.
            const sidebar = rowLabelsRef?.current;
            if (sidebar && sidebar.scrollTop !== container.scrollTop) {
                sidebar.scrollTop = container.scrollTop;
            }

            if (scrollRafRef.current !== null) {
                return;
            }

            scrollRafRef.current = requestAnimationFrame(() => {
                setScrollLeft(container.scrollLeft);
                scrollRafRef.current = null;
            });
        };
        const updateWidth = () => setContainerWidth(container.clientWidth);

        container.addEventListener('scroll', handleScroll, {passive: true});
        const resizeObserver = new ResizeObserver(updateWidth);
        resizeObserver.observe(container);
        updateWidth();
        handleScroll();

        const sidebar = rowLabelsRef?.current;
        const handleSidebarScroll = () => {
            if (!sidebar) return;
            if (container.scrollTop !== sidebar.scrollTop) {
                container.scrollTop = sidebar.scrollTop;
            }
        };
        if (sidebar) {
            sidebar.addEventListener('scroll', handleSidebarScroll, {passive: true});
        }

        return () => {
            container.removeEventListener('scroll', handleScroll);
            resizeObserver.disconnect();
            if (sidebar) {
                sidebar.removeEventListener('scroll', handleSidebarScroll);
            }
            if (scrollRafRef.current !== null) {
                cancelAnimationFrame(scrollRafRef.current);
                scrollRafRef.current = null;
            }
        };
    }, [rowLabelsRef]);

    const handleMiniMapScrollChange = useCallback((newScrollLeft: number) => {
        const container = scrollContainerRef.current;
        if (container) {
            container.scrollLeft = newScrollLeft;
        }
    }, []);

    const handleRulerClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
        const container = scrollContainerRef.current;
        if (!container) return;
        const rect = container.getBoundingClientRect();
        const x = e.clientX - rect.left + container.scrollLeft;
        const time = Math.max(0, Math.min(duration, (x - START_LEFT) / scaleWidth));
        onJumpToTime(TimeUtil.formatTime(time));
        onSelectClip(null);
    }, [duration, scaleWidth, onJumpToTime, onSelectClip]);

    const handleTrackAreaClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
        const target = e.target as HTMLElement;
        // Let clip clicks fall through to their own handlers.
        if (target.closest('.ClipBar')) return;
        const container = scrollContainerRef.current;
        if (!container) return;
        const rect = container.getBoundingClientRect();
        const x = e.clientX - rect.left + container.scrollLeft;
        const time = Math.max(0, Math.min(duration, (x - START_LEFT) / scaleWidth));
        onJumpToTime(TimeUtil.formatTime(time));
        onSelectClip(null);
    }, [duration, scaleWidth, onJumpToTime, onSelectClip]);

    const handleClipClick = useCallback((clipId: string, clipStart: number) => {
        onSelectClip(clipId);
        onJumpToTime(TimeUtil.formatTime(clipStart));
    }, [onSelectClip, onJumpToTime]);

    return (
        <div className="CustomTimelineOuter">
            <div className="CustomTimeline" ref={scrollContainerRef}>
                <div className="TimelineContent" style={{width: Math.max(totalWidth, 100)}}>
                    <div className="RulerArea" onClick={handleRulerClick}>
                        <TimeRuler
                            duration={duration}
                            scaleWidth={scaleWidth}
                            scrollLeft={scrollLeft}
                            startLeft={START_LEFT}
                        />
                    </div>

                    <div className="TrackArea" onClick={handleTrackAreaClick}>
                        {tracks.map(track => (
                            <div className="TrackRow" key={track.id}>
                                {track.clips.map(clip => {
                                    const clipLeft = secondsToPixels(clip.start);
                                    const clipEnd = clip.isOngoing
                                        ? Math.max(currentTime, clip.end)
                                        : clip.end;
                                    const clipRight = secondsToPixels(clipEnd);

                                    // Viewport culling: skip clips entirely outside visible area.
                                    if (clipRight < viewLeft || clipLeft > viewRight) return null;

                                    const clipDuration = Math.max(clipEnd - clip.start, 0.01);
                                    // Natural width = exact pixel span of the clip. Don't inflate it —
                                    // a 4px floor made short clips visibly overshoot their end time.
                                    const naturalWidth = Math.max(clipRight - clipLeft, 1);
                                    const isShort = naturalWidth < 32;
                                    // For short clips the visible mark stays at natural width but the
                                    // button gets padded out to a clickable size. The pad is applied
                                    // symmetrically via CSS var so the colored mark stays centered on
                                    // the true clip range.
                                    const MIN_HIT_WIDTH = 18;
                                    const hitPad = isShort
                                        ? Math.max(0, (MIN_HIT_WIDTH - naturalWidth) / 2)
                                        : 0;
                                    const buttonLeft = clipLeft - hitPad;
                                    const buttonWidth = naturalWidth + hitPad * 2;

                                    const isSelected = clip.id === activeLabelId;

                                    const title = `${clip.labelName} (${clip.start.toFixed(2)}s - ${clip.end.toFixed(2)}s)${clip.isOngoing ? ' - Recording' : ''}`;

                                    return (
                                        <button
                                            key={clip.id}
                                            type="button"
                                            className={classNames('ClipBar', {
                                                ongoing: clip.isOngoing,
                                                selected: isSelected,
                                                short: isShort,
                                            })}
                                            style={{
                                                left: buttonLeft,
                                                width: buttonWidth,
                                                ['--clip-color' as string]: clip.color,
                                                ['--clip-mark-left' as string]: `${hitPad}px`,
                                                ['--clip-mark-width' as string]: `${naturalWidth}px`,
                                                ...(isShort ? {} : {backgroundColor: clip.color}),
                                            }}
                                            title={title}
                                            onClick={() => handleClipClick(clip.id, clip.start)}
                                        >
                                            {isShort ? (
                                                <span className="ClipMark" aria-hidden="true" />
                                            ) : (
                                                <>
                                                    <span className="ActionName">{clip.labelName}</span>
                                                    <span className="ActionMeta">{clipDuration.toFixed(1)}s</span>
                                                </>
                                            )}
                                            {!isShort && (
                                                <ClipHandle
                                                    clipId={clip.id}
                                                    clipStart={clip.start}
                                                    clipEnd={clipEnd}
                                                    clipWidth={naturalWidth}
                                                    scaleWidth={scaleWidth}
                                                    duration={duration}
                                                    tracks={tracks}
                                                    startLeft={START_LEFT}
                                                    onJumpToTime={onJumpToTime}
                                                />
                                            )}
                                        </button>
                                    );
                                })}
                            </div>
                        ))}
                    </div>

                    <Playhead
                        currentTime={currentTime}
                        playheadClock={playheadClock}
                        scaleWidth={scaleWidth}
                        duration={duration}
                        startLeft={START_LEFT}
                        onJumpToTime={onJumpToTime}
                        scrollContainerRef={scrollContainerRef}
                    />
                </div>
            </div>
            <MiniMap
                duration={duration}
                tracks={tracks}
                scrollLeft={scrollLeft}
                containerWidth={containerWidth}
                scaleWidth={scaleWidth}
                startLeft={START_LEFT}
                onScrollChange={handleMiniMapScrollChange}
            />
        </div>
    );
};

export default CustomTimeline;
