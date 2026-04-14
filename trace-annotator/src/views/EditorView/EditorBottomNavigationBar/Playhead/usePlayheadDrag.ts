import {useCallback, useRef, useState} from 'react';

interface UsePlayheadDragOptions {
    duration: number;
    scaleWidth: number;
    startLeft: number;
    onJumpToTime: (time: string) => void;
    formatTime: (seconds: number) => string;
    scrollContainerRef: React.RefObject<HTMLDivElement | null>;
}

interface UsePlayheadDragResult {
    isDragging: boolean;
    handlePointerDown: (e: React.PointerEvent) => void;
}

export function usePlayheadDrag({
    duration,
    scaleWidth,
    startLeft,
    onJumpToTime,
    formatTime,
    scrollContainerRef
}: UsePlayheadDragOptions): UsePlayheadDragResult {
    const [isDragging, setIsDragging] = useState(false);
    const rafRef = useRef<number | null>(null);

    const pixelsToTime = useCallback((clientX: number): number => {
        const container = scrollContainerRef.current;
        if (!container) return 0;
        const rect = container.getBoundingClientRect();
        const x = clientX - rect.left + container.scrollLeft;
        const time = (x - startLeft) / scaleWidth;
        return Math.max(0, Math.min(duration, time));
    }, [scaleWidth, startLeft, duration, scrollContainerRef]);

    const handlePointerDown = useCallback((e: React.PointerEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);

        const target = e.currentTarget as HTMLElement;
        target.setPointerCapture(e.pointerId);

        const onPointerMove = (ev: PointerEvent) => {
            if (rafRef.current !== null) {
                cancelAnimationFrame(rafRef.current);
            }
            rafRef.current = requestAnimationFrame(() => {
                const time = pixelsToTime(ev.clientX);
                onJumpToTime(formatTime(time));
                rafRef.current = null;
            });
        };

        const onPointerUp = () => {
            setIsDragging(false);
            if (rafRef.current !== null) {
                cancelAnimationFrame(rafRef.current);
                rafRef.current = null;
            }
            target.removeEventListener('pointermove', onPointerMove);
            target.removeEventListener('pointerup', onPointerUp);
        };

        target.addEventListener('pointermove', onPointerMove);
        target.addEventListener('pointerup', onPointerUp);
    }, [pixelsToTime, onJumpToTime, formatTime]);

    return {isDragging, handlePointerDown};
}
