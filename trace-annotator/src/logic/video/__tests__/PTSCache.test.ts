import {nearestPtsIndex, floorPtsIndex, snapTime} from '../PTSCache';

describe('nearestPtsIndex', () => {
    it('returns 0 for an empty array', () => {
        expect(nearestPtsIndex(new Float32Array([]), 1.0)).toBe(0);
    });

    it('clamps before the first frame to index 0', () => {
        const pts = new Float32Array([0.0, 0.033, 0.066, 0.1]);
        expect(nearestPtsIndex(pts, -10)).toBe(0);
    });

    it('clamps past the last frame to length-1', () => {
        const pts = new Float32Array([0.0, 0.033, 0.066, 0.1]);
        expect(nearestPtsIndex(pts, 99)).toBe(pts.length - 1);
    });

    it('rounds to the nearer neighbor on a CFR grid', () => {
        const pts = new Float32Array([0.0, 0.033, 0.066, 0.1]);
        // Closer to 0.033 (delta=0.001) than 0.066 (delta=0.032).
        expect(nearestPtsIndex(pts, 0.034)).toBe(1);
        // Closer to 0.066.
        expect(nearestPtsIndex(pts, 0.065)).toBe(2);
    });

    it('handles non-uniform PTS spacing (VFR)', () => {
        // Synthetic VFR: a 50 ms gap right after frame 1.
        const pts = new Float32Array([0.0, 0.033, 0.083, 0.116]);
        // 0.06 is closer to 0.083 (Δ=0.023) than 0.033 (Δ=0.027).
        expect(nearestPtsIndex(pts, 0.06)).toBe(2);
        // 0.05 is closer to 0.033 (Δ=0.017) than 0.083 (Δ=0.033).
        expect(nearestPtsIndex(pts, 0.05)).toBe(1);
    });
});

describe('floorPtsIndex', () => {
    it('returns 0 for times at or before the first frame', () => {
        const pts = new Float32Array([1.0, 1.5, 2.0]);
        expect(floorPtsIndex(pts, 0.0)).toBe(0);
        expect(floorPtsIndex(pts, 1.0)).toBe(0);
    });

    it('returns the index of the currently displayed frame (floor)', () => {
        const pts = new Float32Array([0.0, 0.033, 0.066, 0.1]);
        // Frame 1 is on screen for [0.033, 0.066). 0.065 → still frame 1.
        expect(floorPtsIndex(pts, 0.065)).toBe(1);
        // Exactly at frame 2's PTS → frame 2.
        expect(floorPtsIndex(pts, 0.066)).toBe(2);
    });

    it('clamps past the last frame', () => {
        const pts = new Float32Array([0.0, 0.033, 0.066, 0.1]);
        expect(floorPtsIndex(pts, 99)).toBe(pts.length - 1);
    });
});

describe('snapTime', () => {
    it('uses PTS when supplied', () => {
        const pts = new Float32Array([0.0, 0.05, 0.13]);
        const r = snapTime(0.07, pts, 30);
        // 0.07 is closer to 0.05 (Δ=0.02) than 0.13 (Δ=0.06).
        expect(r.frame).toBe(1);
        expect(r.time).toBeCloseTo(0.05, 5);
    });

    it('falls back to nominal-fps grid when PTS is null', () => {
        // Off-grid time at 30 fps: 0.034 → frame 1 (1/30 ≈ 0.0333).
        const r = snapTime(0.034, null, 30);
        expect(r.frame).toBe(1);
        expect(r.time).toBeCloseTo(1 / 30, 5);
    });

    it('falls back when PTS array is empty', () => {
        const r = snapTime(0.5, new Float32Array([]), 30);
        expect(r.frame).toBe(15);
        expect(r.time).toBeCloseTo(0.5, 5);
    });

    it('clamps negative time to frame 0 in fallback', () => {
        const r = snapTime(-1.5, null, 30);
        expect(r.frame).toBe(0);
        expect(r.time).toBe(0);
    });
});
