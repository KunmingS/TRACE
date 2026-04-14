import React from 'react';
import './Playhead.scss';
import {usePlayheadDrag} from './usePlayheadDrag';
import {TimeUtil} from '../../../../utils/TimeUtil';

interface PlayheadProps {
    currentTime: number;
    scaleWidth: number;
    duration: number;
    startLeft: number;
    onJumpToTime: (time: string) => void;
    scrollContainerRef: React.RefObject<HTMLDivElement | null>;
}

const Playhead: React.FC<PlayheadProps> = ({
    currentTime,
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

    const left = startLeft + currentTime * scaleWidth;

    return (
        <div
            className="Playhead"
            style={{left}}
        >
            <div
                className={`Playhead__handle ${isDragging ? 'Playhead__handle--dragging' : ''}`}
                onPointerDown={handlePointerDown}
            >
                <svg
                    className="Playhead__triangle"
                    width="14"
                    height="10"
                    viewBox="0 0 14 10"
                >
                    <path d="M0 0 L14 0 L7 10 Z" fill="#3b9eff" />
                </svg>
            </div>
            <div className="Playhead__line" />
        </div>
    );
};

export default React.memo(Playhead);
