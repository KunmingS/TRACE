import {useMemo} from 'react';

const INTERVALS = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
const MIN_MAJOR_TICK_PX = 60;
const MINOR_DIVISIONS = 5;

export interface Tick {
    time: number;
    x: number;
    isMajor: boolean;
    label: string | null;
}

function formatTimecode(seconds: number): string {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;

    if (h > 0) {
        return `${h}:${String(m).padStart(2, '0')}:${String(Math.floor(s)).padStart(2, '0')}`;
    }
    if (m > 0) {
        // Show fractional seconds only for sub-second intervals
        const sStr = s === Math.floor(s)
            ? String(Math.floor(s)).padStart(2, '0')
            : s.toFixed(1).padStart(4, '0');
        return `${m}:${sStr}`;
    }
    // Seconds only
    if (seconds === Math.floor(seconds)) {
        return `${Math.floor(seconds)}s`;
    }
    return `${seconds.toFixed(1)}s`;
}

export function useTickGenerator(
    duration: number,
    scaleWidth: number,
    startLeft: number
): Tick[] {
    return useMemo(() => {
        if (duration <= 0 || scaleWidth <= 0) return [];

        // Pick the smallest interval where major ticks are at least MIN_MAJOR_TICK_PX apart
        let majorInterval = INTERVALS[INTERVALS.length - 1];
        for (const interval of INTERVALS) {
            if (interval * scaleWidth >= MIN_MAJOR_TICK_PX) {
                majorInterval = interval;
                break;
            }
        }

        const minorInterval = majorInterval / MINOR_DIVISIONS;
        const maxTime = duration;
        const ticks: Tick[] = [];

        // Generate ticks from 0 up to duration
        // Use minor interval as the stepping unit to catch both major and minor ticks
        const totalSteps = Math.ceil(maxTime / minorInterval);

        for (let i = 0; i <= totalSteps; i++) {
            const time = +(i * minorInterval).toFixed(6);
            if (time > maxTime + minorInterval * 0.01) break;

            const x = startLeft + time * scaleWidth;

            // Determine if this is a major tick: time is a multiple of majorInterval
            // Use a small epsilon for floating point comparison
            const remainder = time % majorInterval;
            const isMajor = remainder < 1e-6 || (majorInterval - remainder) < 1e-6;

            ticks.push({
                time,
                x,
                isMajor,
                label: isMajor ? formatTimecode(time) : null
            });
        }

        return ticks;
    }, [duration, scaleWidth, startLeft]);
}
