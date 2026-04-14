import { v4 as uuidv4 } from 'uuid';
import { ImageData, LabelName, LabelRect } from '../store/labels/types';
import { LabelStatus } from '../data/enums/LabelStatus';
import { TimeUtil } from './TimeUtil';

export function toggleBehaviorClip(
    labelName: LabelName,
    imageData: ImageData,
    currentTime: number,
    frameRate: number
): ImageData {
    const { formattedTime } = TimeUtil.formatTimeWithFrame(currentTime, frameRate);

    const unfinishedRect = imageData.labelRects?.find(rect =>
        rect.labelId === labelName.id && !rect.endTimestamp
    );

    if (unfinishedRect) {
        const updatedLabelRects = imageData.labelRects.map(rect => {
            if (rect.id === unfinishedRect.id) {
                return { ...rect, endTimestamp: formattedTime };
            }
            return rect;
        });
        return { ...imageData, labelRects: updatedLabelRects };
    } else {
        const tsStr = currentTime.toFixed(3) + 's';
        const newLabelRect: LabelRect = {
            id: uuidv4(),
            labelId: labelName.id,
            isVisible: true,
            isCreatedByAI: false,
            status: LabelStatus.ACCEPTED,
            suggestedLabel: null,
            timestamp: tsStr,
            endTimestamp: undefined,
            behavior: labelName.name
        };
        return {
            ...imageData,
            labelRects: [...(imageData.labelRects || []), newLabelRect]
        };
    }
}
