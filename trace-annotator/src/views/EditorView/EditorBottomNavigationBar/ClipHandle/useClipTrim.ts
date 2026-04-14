import {useCallback, useRef, useState} from 'react';
import {store} from '../../../..';
import {updateImageDataById} from '../../../../store/labels/actionCreators';
import {ImageData, LabelRect} from '../../../../store/labels/types';
import {TimeUtil} from '../../../../utils/TimeUtil';

export type TrimSide = 'left' | 'right' | null;

interface UseClipTrimOptions {
    clipId: string;
    clipStart: number;
    clipEnd: number;
    scaleWidth: number;
    duration: number;
    onJumpToTime: (time: string) => void;
}

interface UseClipTrimResult {
    trimSide: TrimSide;
    isDragging: boolean;
    handlePointerDown: (side: 'left' | 'right', e: React.PointerEvent) => void;
}

const MIN_CLIP_DURATION = 0.1;

export function useClipTrim({
    clipId,
    clipStart,
    clipEnd,
    scaleWidth,
    duration,
    onJumpToTime,
}: UseClipTrimOptions): UseClipTrimResult {
    const [trimSide, setTrimSide] = useState<TrimSide>(null);
    const [isDragging, setIsDragging] = useState(false);
    const dragRef = useRef<{
        side: 'left' | 'right';
        startX: number;
        originalStart: number;
        originalEnd: number;
    } | null>(null);

    const commitTrim = useCallback((newStart: number, newEnd: number) => {
        const state = store.getState();
        const {imagesData, activeImageIndex} = state.labels;
        const imageData: ImageData = imagesData[activeImageIndex];
        if (!imageData) return;

        const updatedRects: LabelRect[] = imageData.labelRects.map(rect => {
            if (rect.id !== clipId) return rect;
            const {formattedTime: startTime, frame: startFrame} =
                TimeUtil.formatTimeWithFrame(newStart, imageData.frameRate || 30);
            const {formattedTime: endTime, frame: endFrame} =
                TimeUtil.formatTimeWithFrame(newEnd, imageData.frameRate || 30);
            return {
                ...rect,
                timestamp: startTime,
                frame: startFrame,
                endTimestamp: endTime,
                endFrame: endFrame,
            };
        });

        store.dispatch(
            updateImageDataById(imageData.id, {
                ...imageData,
                labelRects: updatedRects,
            })
        );
    }, [clipId]);

    const handlePointerDown = useCallback(
        (side: 'left' | 'right', e: React.PointerEvent) => {
            e.stopPropagation();
            e.preventDefault();

            dragRef.current = {
                side,
                startX: e.clientX,
                originalStart: clipStart,
                originalEnd: clipEnd,
            };
            setTrimSide(side);
            setIsDragging(true);

            const target = e.currentTarget as HTMLElement;
            target.setPointerCapture(e.pointerId);

            const onPointerMove = (ev: PointerEvent) => {
                const drag = dragRef.current;
                if (!drag) return;

                const deltaX = ev.clientX - drag.startX;
                const deltaTime = deltaX / scaleWidth;

                let newStart = drag.originalStart;
                let newEnd = drag.originalEnd;

                if (drag.side === 'left') {
                    newStart = Math.max(0, drag.originalStart + deltaTime);
                    if (newEnd - newStart < MIN_CLIP_DURATION) {
                        newStart = newEnd - MIN_CLIP_DURATION;
                    }
                } else {
                    newEnd = Math.min(duration, drag.originalEnd + deltaTime);
                    if (newEnd - newStart < MIN_CLIP_DURATION) {
                        newEnd = newStart + MIN_CLIP_DURATION;
                    }
                }

                commitTrim(newStart, newEnd);

                // Seek video to the edge being dragged
                const seekTime = drag.side === 'left' ? newStart : newEnd;
                onJumpToTime(TimeUtil.formatTime(seekTime));
            };

            const onPointerUp = () => {
                dragRef.current = null;
                setTrimSide(null);
                setIsDragging(false);
                window.removeEventListener('pointermove', onPointerMove);
                window.removeEventListener('pointerup', onPointerUp);
            };

            window.addEventListener('pointermove', onPointerMove);
            window.addEventListener('pointerup', onPointerUp);
        },
        [clipStart, clipEnd, scaleWidth, duration, commitTrim, onJumpToTime]
    );

    return {trimSide, isDragging, handlePointerDown};
}
