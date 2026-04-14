export interface TimelineClip {
    id: string;
    start: number;      // seconds
    end: number;        // seconds
    labelId: string;
    color: string;
    labelName: string;
    isOngoing: boolean;
}

export interface TimelineTrack {
    id: string;
    name: string;
    color: string;
    clips: TimelineClip[];
}
