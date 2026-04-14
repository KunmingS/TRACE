import {useEffect, useRef, RefObject} from 'react';

const THRESHOLD = 5;
const START_LEFT = 20;

export function useAutoScroll(
    currentTime: number,
    duration: number,
    scaleWidth: number,
    scrollContainerRef: RefObject<HTMLDivElement>
) {
    const lastScrollRef = useRef(-1);

    useEffect(() => {
        const container = scrollContainerRef.current;
        if (!container) return;

        const containerWidth = container.clientWidth;
        const totalWidth = START_LEFT + duration * scaleWidth;
        let scrollLeft = 0;

        if (currentTime <= THRESHOLD) {
            scrollLeft = 0;
        } else if (currentTime >= duration - THRESHOLD) {
            scrollLeft = Math.max(0, totalWidth - containerWidth);
        } else {
            scrollLeft = START_LEFT + currentTime * scaleWidth - containerWidth / 2;
            scrollLeft = Math.max(0, scrollLeft);
        }

        if (Math.abs(scrollLeft - lastScrollRef.current) > 1) {
            container.scrollLeft = scrollLeft;
            lastScrollRef.current = scrollLeft;
        }
    }, [currentTime, duration, scaleWidth, scrollContainerRef]);
}
