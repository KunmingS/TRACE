export interface PlayheadClock {
    current: number;
}

export const createPlayheadClock = (): PlayheadClock => ({current: 0});
