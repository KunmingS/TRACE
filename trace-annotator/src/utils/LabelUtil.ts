import {Annotation, LabelName, LabelRect} from '../store/labels/types';
import { v4 as uuidv4 } from 'uuid';
import {find} from 'lodash';
import {LabelStatus} from '../data/enums/LabelStatus';
import { sample } from 'lodash';
import {Settings} from '../settings/Settings';

export class LabelUtil {
    public static createLabelName(name: string): LabelName {
        return {
            id: uuidv4(),
            name,
            color: sample(Settings.LABEL_COLORS_PALETTE)
        }
    }

    public static createLabelRect(labelId: string, startFrame: number, endFrame?: number, behavior?: string, imageData?: any): LabelRect {
        const fr = imageData?.frameRate ?? 30;
        const ts = (startFrame / fr).toFixed(3) + 's';
        const endTs = typeof endFrame === 'number'
            ? (endFrame / fr).toFixed(3) + 's'
            : undefined;
        return {
            id: uuidv4(),
            labelId,
            isVisible: true,
            isCreatedByAI: false,
            status: LabelStatus.ACCEPTED,
            suggestedLabel: null,
            timestamp: ts,
            endTimestamp: endTs,
            behavior
        };
    }

    public static toggleAnnotationVisibility<AnnotationType extends Annotation>(annotation: AnnotationType): AnnotationType {
        return {
            ...annotation,
            isVisible: !annotation.isVisible
        }
    }

    public static labelNamesIdsDiff(oldLabelNames: LabelName[], newLabelNames: LabelName[]): string[] {
        return oldLabelNames.reduce((missingIds: string[], labelName: LabelName) => {
            if (!find(newLabelNames, { 'id': labelName.id })) {
                missingIds.push(labelName.id);
            }
            return missingIds
        }, [])
    }
}
