export interface TimelineClip {
    id: string;
    start: number;      // seconds
    end: number;        // seconds
    labelId: string;
    color: string;
    labelName: string;
    isOngoing: boolean;
    animalId?: string;
}

export interface TimelineTrack {
    id: string;
    name: string;
    color: string;
    clips: TimelineClip[];
}
