import { v4 as uuidv4 } from 'uuid';
import { ImageData, LabelName, LabelRect } from '../store/labels/types';
import { LabelStatus } from '../data/enums/LabelStatus';
import { TimeUtil } from './TimeUtil';
import { snapTime, nearestPtsIndex } from '../logic/video/PTSCache';
import { DEFAULT_SUBJECT_ID } from '../store/labels/reducer';

// Snap the playhead to the nearest real frame (PTS-aware) and persist that
// frame's timestamp. On open: a clip starts at frame N. On close: the end
// is forced to be at least one frame past the start so the resulting
// interval covers ≥ 1 frame — otherwise short presses produce "0 fr"
// clips that silently disappear from training data.
export function toggleBehaviorClip(
    labelName: LabelName,
    imageData: ImageData,
    currentTime: number,
    frameRate: number,
    activeSubjectId: string | null,
    frameTimestamps: Float32Array | null = null,
): ImageData {
    const subjectId = activeSubjectId ?? DEFAULT_SUBJECT_ID;
    const snapped = snapTime(currentTime, frameTimestamps, frameRate);

    const unfinishedRect = imageData.labelRects?.find(rect =>
        rect.labelId === labelName.id
        && (rect.animalId ?? DEFAULT_SUBJECT_ID) === subjectId
        && !rect.endTimestamp
    );

    if (unfinishedRect) {
        // Closing an open clip: bump the end forward if it landed on or
        // before the start frame, so the saved interval is ≥ 1 frame.
        // Without this guard, a quick double-press on the same frame
        // produces a zero-length clip.
        let startFrame: number | null = null;
        if (typeof unfinishedRect.frame === 'number') {
            startFrame = unfinishedRect.frame;
        } else if (unfinishedRect.timestamp) {
            const startSec = TimeUtil.parseTimestamp(unfinishedRect.timestamp);
            startFrame = frameTimestamps && frameTimestamps.length > 0
                ? nearestPtsIndex(frameTimestamps, startSec)
                : Math.max(0, Math.round(startSec * (frameRate || 30)));
        }
        let endFrame = snapped.frame;
        let endTime = snapped.time;
        if (startFrame != null && endFrame <= startFrame) {
            const bumped = startFrame + 1;
            if (frameTimestamps && bumped < frameTimestamps.length) {
                endFrame = bumped;
                endTime = frameTimestamps[bumped];
            } else {
                endFrame = bumped;
                endTime = bumped / (frameRate || 30);
            }
        }
        const { formattedTime } = TimeUtil.formatTimeWithFrame(endTime, frameRate);
        const updatedLabelRects = imageData.labelRects.map(rect => {
            if (rect.id === unfinishedRect.id) {
                return { ...rect, endTimestamp: formattedTime, endFrame };
            }
            return rect;
        });
        return { ...imageData, labelRects: updatedLabelRects };
    } else {
        const { formattedTime } = TimeUtil.formatTimeWithFrame(snapped.time, frameRate);
        const newLabelRect: LabelRect = {
            id: uuidv4(),
            labelId: labelName.id,
            isVisible: true,
            isCreatedByAI: false,
            status: LabelStatus.ACCEPTED,
            suggestedLabel: null,
            timestamp: formattedTime,
            frame: snapped.frame,
            endTimestamp: undefined,
            behavior: labelName.name,
            animalId: subjectId
        };
        return {
            ...imageData,
            labelRects: [...(imageData.labelRects || []), newLabelRect]
        };
    }
}
