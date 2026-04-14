import {useMemo} from 'react';

const START_LEFT = 20;

export function useTimelineLayout(
    duration: number,
    zoom: number,
    containerWidth: number,
    isCompactLayout: boolean
) {
    const scaleWidth = Math.max(
        80,
        Math.min(
            160,
            (isCompactLayout ? 80 : 90) + Math.max(0, zoom - 1) * 24
        )
    );

    const totalWidth = START_LEFT + duration * scaleWidth;

    const secondsToPixels = useMemo(() => {
        return (t: number): number => START_LEFT + t * scaleWidth;
    }, [scaleWidth]);

    const pixelsToSeconds = useMemo(() => {
        return (x: number): number => Math.max(0, (x - START_LEFT) / scaleWidth);
    }, [scaleWidth]);

    return {totalWidth, secondsToPixels, pixelsToSeconds, scaleWidth};
}
