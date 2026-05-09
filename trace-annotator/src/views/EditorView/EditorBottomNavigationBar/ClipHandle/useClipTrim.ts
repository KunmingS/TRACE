import {useCallback, useRef, useState} from 'react';
import {store} from '../../../..';
import {updateImageDataById} from '../../../../store/labels/actionCreators';
import {ImageData, LabelRect} from '../../../../store/labels/types';
import {TimeUtil} from '../../../../utils/TimeUtil';
import {getPtsIfReady, snapTime} from '../../../../logic/video/PTSCache';

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

        // PTSCache is keyed on (dir, filename); both stamped onto imageData
        // by FileBrowser.playFile. If the PTS fetch hasn't resolved yet
        // (rare — only during the first 2–5 s of a cold-cache video),
        // ``getPtsIfReady`` returns null and ``snapTime`` falls back to
        // nominal-fps math, which is what the trim handle did pre-PTS.
        const dir = (imageData as any).videoPath as string | undefined;
        const filename = imageData.fileData?.name;
        const pts = (dir && filename) ? getPtsIfReady(filename, dir) : null;
        const fps = imageData.frameRate || 30;

        const start = snapTime(newStart, pts, fps);
        let end = snapTime(newEnd, pts, fps);
        // Force the snapped interval to span ≥ 1 frame. Without this the
        // handle could collapse onto its sibling — same frame on both
        // sides — which the timeline renders but the dataset effectively
        // drops.
        if (end.frame <= start.frame) {
            const bumped = start.frame + 1;
            if (pts && bumped < pts.length) {
                end = { time: pts[bumped], frame: bumped };
            } else {
                end = { time: bumped / fps, frame: bumped };
            }
        }

        const updatedRects: LabelRect[] = imageData.labelRects.map(rect => {
            if (rect.id !== clipId) return rect;
            const {formattedTime: startTime} =
                TimeUtil.formatTimeWithFrame(start.time, fps);
            const {formattedTime: endTime} =
                TimeUtil.formatTimeWithFrame(end.time, fps);
            return {
                ...rect,
                timestamp: startTime,
                frame: start.frame,
                endTimestamp: endTime,
                endFrame: end.frame,
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

                // One-frame floor: prevents start==end (zero-width clip) while
                // still letting users author the tightest possible annotation.
                const {imagesData, activeImageIndex} = store.getState().labels;
                const fps = imagesData[activeImageIndex]?.frameRate || 30;
                const minDuration = 1 / fps;

                let newStart = drag.originalStart;
                let newEnd = drag.originalEnd;

                if (drag.side === 'left') {
                    newStart = Math.max(0, drag.originalStart + deltaTime);
                    if (newEnd - newStart < minDuration) {
                        newStart = newEnd - minDuration;
                    }
                } else {
                    newEnd = Math.min(duration, drag.originalEnd + deltaTime);
                    if (newEnd - newStart < minDuration) {
                        newEnd = newStart + minDuration;
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
