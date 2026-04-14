import {useMemo} from 'react';
import {TimelineTrack} from '../CustomTimeline/types';

interface SnapTarget {
    time: number;
    isActive: boolean;
}

interface UseSnapIndicatorOptions {
    clipId: string;
    clipStart: number;
    clipEnd: number;
    tracks: TimelineTrack[];
    scaleWidth: number;
    isDragging: boolean;
    trimSide: 'left' | 'right' | null;
}

const SNAP_THRESHOLD_PX = 8;

export function useSnapIndicator({
    clipId,
    clipStart,
    clipEnd,
    tracks,
    scaleWidth,
    isDragging,
    trimSide,
}: UseSnapIndicatorOptions): SnapTarget {
    return useMemo(() => {
        if (!isDragging || !trimSide) {
            return {time: 0, isActive: false};
        }

        const edgeTime = trimSide === 'left' ? clipStart : clipEnd;
        const thresholdTime = SNAP_THRESHOLD_PX / scaleWidth;

        let closestDistance = Infinity;
        let closestTime = 0;

        for (const track of tracks) {
            for (const clip of track.clips) {
                if (clip.id === clipId) continue;

                const edges = [clip.start, clip.end];
                for (const edge of edges) {
                    const distance = Math.abs(edge - edgeTime);
                    if (distance < closestDistance && distance <= thresholdTime) {
                        closestDistance = distance;
                        closestTime = edge;
                    }
                }
            }
        }

        if (closestDistance <= thresholdTime) {
            return {time: closestTime, isActive: true};
        }

        return {time: 0, isActive: false};
    }, [clipId, clipStart, clipEnd, tracks, scaleWidth, isDragging, trimSide]);
}
