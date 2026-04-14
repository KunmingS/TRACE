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

interface CustomTimelineProps {
    tracks: TimelineTrack[];
    currentTime: number;
    duration: number;
    scaleWidth: number;
    onJumpToTime: (time: string) => void;
    isCompactLayout: boolean;
}

const START_LEFT = 20;

const CustomTimeline: React.FC<CustomTimelineProps> = ({
    tracks,
    currentTime,
    duration,
    scaleWidth,
    onJumpToTime,
    isCompactLayout
}) => {
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const [scrollLeft, setScrollLeft] = useState(0);
    const [containerWidth, setContainerWidth] = useState(0);

    const totalWidth = START_LEFT + duration * scaleWidth;

    const secondsToPixels = (t: number): number => START_LEFT + t * scaleWidth;

    useAutoScroll(currentTime, duration, scaleWidth, scrollContainerRef);

    useEffect(() => {
        const container = scrollContainerRef.current;
        if (!container) return undefined;

        const handleScroll = () => setScrollLeft(container.scrollLeft);
        const updateWidth = () => setContainerWidth(container.clientWidth);

        container.addEventListener('scroll', handleScroll, {passive: true});
        const resizeObserver = new ResizeObserver(updateWidth);
        resizeObserver.observe(container);
        updateWidth();
        handleScroll();

        return () => {
            container.removeEventListener('scroll', handleScroll);
            resizeObserver.disconnect();
        };
    }, []);

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
    }, [duration, scaleWidth, onJumpToTime]);

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

                    <div className="TrackArea">
                        {tracks.map(track => (
                            <div className="TrackRow" key={track.id}>
                                {track.clips.map(clip => {
                                    const clipLeft = secondsToPixels(clip.start);
                                    const clipRight = secondsToPixels(clip.end);
                                    // Viewport culling: skip clips entirely outside visible area
                                    const viewLeft = scrollLeft - 200;
                                    const viewRight = scrollLeft + containerWidth + 200;
                                    if (clipRight < viewLeft || clipLeft > viewRight) return null;

                                    const clipDuration = Math.max(clip.end - clip.start, 0.1);
                                    const clipWidth = Math.max(clipRight - clipLeft, 4);

                                    return (
                                        <button
                                            key={clip.id}
                                            type="button"
                                            className={classNames('ClipBar', {ongoing: clip.isOngoing})}
                                            style={{
                                                left: clipLeft,
                                                width: clipWidth,
                                                backgroundColor: clip.color,
                                            }}
                                            title={`${clip.labelName} (${clip.start.toFixed(2)}s - ${clip.end.toFixed(2)}s)${clip.isOngoing ? ' - Recording' : ''}`}
                                            onClick={() => onJumpToTime(TimeUtil.formatTime(clip.start))}
                                        >
                                            <span className="ActionName">{clip.labelName}</span>
                                            <span className="ActionMeta">{clipDuration.toFixed(1)}s</span>
                                            <ClipHandle
                                                clipId={clip.id}
                                                clipStart={clip.start}
                                                clipEnd={clip.end}
                                                clipWidth={clipWidth}
                                                scaleWidth={scaleWidth}
                                                duration={duration}
                                                tracks={tracks}
                                                startLeft={START_LEFT}
                                                onJumpToTime={onJumpToTime}
                                            />
                                        </button>
                                    );
                                })}
                            </div>
                        ))}
                    </div>

                    <Playhead
                        currentTime={currentTime}
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
