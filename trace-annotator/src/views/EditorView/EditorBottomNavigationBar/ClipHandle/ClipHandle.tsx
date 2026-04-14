import React from 'react';
import './ClipHandle.scss';
import {useClipTrim} from './useClipTrim';
import {useSnapIndicator} from './useSnapIndicator';
import {TimelineTrack} from '../CustomTimeline/types';

interface ClipHandleProps {
    clipId: string;
    clipStart: number;
    clipEnd: number;
    clipWidth: number;
    scaleWidth: number;
    duration: number;
    tracks: TimelineTrack[];
    startLeft: number;
    onJumpToTime: (time: string) => void;
}

const ClipHandle: React.FC<ClipHandleProps> = ({
    clipId,
    clipStart,
    clipEnd,
    clipWidth,
    scaleWidth,
    duration,
    tracks,
    startLeft,
    onJumpToTime,
}) => {
    const {trimSide, isDragging, handlePointerDown} = useClipTrim({
        clipId,
        clipStart,
        clipEnd,
        scaleWidth,
        duration,
        onJumpToTime,
    });

    const snap = useSnapIndicator({
        clipId,
        clipStart,
        clipEnd,
        tracks,
        scaleWidth,
        isDragging,
        trimSide,
    });

    // Snap indicator position relative to the clip's left edge
    const clipLeftPx = startLeft + clipStart * scaleWidth;
    const snapOffsetPx = snap.isActive
        ? (startLeft + snap.time * scaleWidth) - clipLeftPx
        : 0;

    return (
        <>
            <div
                className="ClipHandleZone ClipHandleLeft"
                onPointerDown={(e) => handlePointerDown('left', e)}
            />
            <div
                className="ClipHandleZone ClipHandleRight"
                onPointerDown={(e) => handlePointerDown('right', e)}
            />
            {snap.isActive && (
                <div
                    className="SnapIndicator"
                    style={{left: snapOffsetPx}}
                />
            )}
        </>
    );
};

export default ClipHandle;
