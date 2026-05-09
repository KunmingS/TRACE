import React, {useEffect, useRef} from 'react';
import './Playhead.scss';
import {usePlayheadDrag} from './usePlayheadDrag';
import {TimeUtil} from '../../../../utils/TimeUtil';
import {PlayheadClock} from '../playheadClock';

interface PlayheadProps {
    currentTime: number;
    playheadClock?: PlayheadClock;
    scaleWidth: number;
    duration: number;
    startLeft: number;
    onJumpToTime: (time: string) => void;
    scrollContainerRef: React.RefObject<HTMLDivElement | null>;
}

const Playhead: React.FC<PlayheadProps> = ({
    currentTime,
    playheadClock,
    scaleWidth,
    duration,
    startLeft,
    onJumpToTime,
    scrollContainerRef
}) => {
    const {isDragging, handlePointerDown} = usePlayheadDrag({
        duration,
        scaleWidth,
        startLeft,
        onJumpToTime,
        formatTime: TimeUtil.formatTime,
        scrollContainerRef
    });

    const rootRef = useRef<HTMLDivElement | null>(null);

    // Imperative, rAF-driven positioning. Reads directly from the shared
    // PlayheadClock (mutable ref object) so Plyr timeupdate → playhead motion
    // never goes through React's reconciler.
    useEffect(() => {
        const el = rootRef.current;
        if (!el) return undefined;

        const applyTransform = (t: number) => {
            const x = startLeft + t * scaleWidth;
            el.style.transform = `translate3d(${x}px, 0, 0) translateX(-50%)`;
        };

        // Paint immediately so the initial position is correct even before
        // the first rAF tick, and when the clock isn't wired up.
        applyTransform(playheadClock ? playheadClock.current : currentTime);

        if (!playheadClock) return undefined;

        let raf = 0;
        let last = -1;
        const tick = () => {
            const t = playheadClock.current;
            if (t !== last) {
                applyTransform(t);
                last = t;
            }
            raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);

        return () => cancelAnimationFrame(raf);
    }, [playheadClock, scaleWidth, startLeft, currentTime]);

    return (
        <div
            ref={rootRef}
            className='Playhead'
        >
            <div
                className={`Playhead__handle ${isDragging ? 'Playhead__handle--dragging' : ''}`}
                onPointerDown={handlePointerDown}
            >
                <svg
                    className='Playhead__triangle'
                    width='14'
                    height='10'
                    viewBox='0 0 14 10'
                >
                    <path d='M0 0 L14 0 L7 10 Z' fill='#3b9eff' />
                </svg>
            </div>
            <div className='Playhead__line' />
        </div>
    );
};

export default React.memo(Playhead);
