import {toggleBehaviorClip} from '../BehaviorUtil';
import {ImageData, LabelName, LabelRect} from '../../store/labels/types';
import {LabelStatus} from '../../data/enums/LabelStatus';

const makeLabel = (id: string, name: string): LabelName => ({
    id,
    name,
    color: '#000',
    shortcut: name[0],
});

const makeImage = (rects: LabelRect[] = []): ImageData => ({
    id: 'img-1',
    fileData: new File([], 'video.mp4'),
    loadStatus: true,
    labelRects: rects,
    labelNameIds: [],
    frameRate: 30,
});

describe('toggleBehaviorClip with PTS snapping', () => {
    it('opens a new clip whose timestamp lands on the nearest PTS frame', () => {
        // 30 fps PTS: 0, 0.033, 0.067, 0.1, ...
        const pts = new Float32Array([0, 1/30, 2/30, 3/30, 4/30, 5/30, 6/30]);
        const label = makeLabel('lbl', 'eat');
        const image = makeImage();

        // Press at t = 0.06 → closest is frame 2 (t=0.0667).
        const out = toggleBehaviorClip(label, image, 0.06, 30, null, pts);

        const rect = out.labelRects[0];
        expect(rect.frame).toBe(2);
        // Timestamp string carries the snapped time, not the raw input.
        expect(rect.timestamp).toBe('00:00:00.066');
        expect(rect.endTimestamp).toBeUndefined();
    });

    it('closes a clip with a frame index strictly greater than the start', () => {
        // CFR-shaped PTS so the legacy behavior is unambiguous.
        const pts = new Float32Array([0, 1/30, 2/30, 3/30, 4/30, 5/30, 6/30]);
        const label = makeLabel('lbl', 'eat');
        const openRect: LabelRect = {
            id: 'r1',
            labelId: 'lbl',
            isVisible: true,
            isCreatedByAI: false,
            status: LabelStatus.ACCEPTED,
            suggestedLabel: null,
            timestamp: '00:00:00.066',
            frame: 2,
            endTimestamp: undefined,
            behavior: 'eat',
            animalId: null,
        };
        const image = makeImage([openRect]);

        // Close at t = 0.13 → frame 4 (t≈0.1333).
        const out = toggleBehaviorClip(label, image, 0.13, 30, null, pts);
        const closed = out.labelRects[0];
        expect(closed.endFrame).toBe(4);
        expect(closed.endTimestamp).toBe('00:00:00.133');
    });

    it('bumps the end forward by one frame when the press lands on the start frame', () => {
        // Same fixture; user double-presses on the SAME frame as the start.
        const pts = new Float32Array([0, 1/30, 2/30, 3/30, 4/30]);
        const label = makeLabel('lbl', 'eat');
        const openRect: LabelRect = {
            id: 'r1',
            labelId: 'lbl',
            isVisible: true,
            isCreatedByAI: false,
            status: LabelStatus.ACCEPTED,
            suggestedLabel: null,
            timestamp: '00:00:00.066',
            frame: 2,
            endTimestamp: undefined,
            behavior: 'eat',
            animalId: null,
        };
        const image = makeImage([openRect]);

        // Press exactly at start's PTS → snap → frame 2 → would be zero-length.
        // Should bump to frame 3 instead.
        const out = toggleBehaviorClip(label, image, 2/30, 30, null, pts);
        const closed = out.labelRects[0];
        expect(closed.endFrame).toBe(3);
    });

    it('falls back to nominal-fps math when PTS is null', () => {
        const label = makeLabel('lbl', 'eat');
        const image = makeImage();

        // 0.034 → round(0.034 * 30) = 1 → t = 1/30 ≈ 0.0333.
        const out = toggleBehaviorClip(label, image, 0.034, 30, null, null);
        const rect = out.labelRects[0];
        expect(rect.frame).toBe(1);
        expect(rect.timestamp).toBe('00:00:00.033');
    });
});
