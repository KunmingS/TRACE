import {LabelsActionTypes, ImageData, LabelName, Subject} from './types';
import {Action} from '../Actions';
import {LabelType} from '../../data/enums/LabelType';

export function updateActiveImageIndex(activeImageIndex: number): LabelsActionTypes {
    return {
        type: Action.UPDATE_ACTIVE_IMAGE_INDEX,
        payload: {
            activeImageIndex,
        },
    };
}

export function updateActiveLabelNameId(activeLabelNameId: string): LabelsActionTypes {
    return {
        type: Action.UPDATE_ACTIVE_LABEL_NAME_ID,
        payload: {
            activeLabelNameId,
        },
    };
}

export function updateActiveLabelId(activeLabelId: string): LabelsActionTypes {
    return {
        type: Action.UPDATE_ACTIVE_LABEL_ID,
        payload: {
            activeLabelId,
        },
    };
}

export function updateHighlightedLabelId(highlightedLabelId: string): LabelsActionTypes {
    return {
        type: Action.UPDATE_HIGHLIGHTED_LABEL_ID,
        payload: {
            highlightedLabelId,
        },
    };
}

export function updateActiveLabelType(activeLabelType: LabelType): LabelsActionTypes {
    return {
        type: Action.UPDATE_ACTIVE_LABEL_TYPE,
        payload: {
            activeLabelType,
        },
    };
}

export function updateImageDataById(id: string, newImageData: ImageData): LabelsActionTypes {
    return {
        type: Action.UPDATE_IMAGE_DATA_BY_ID,
        payload: {
            id,
            newImageData
        },
    };
}

export function addImageData(imageData: ImageData[]): LabelsActionTypes {
    return {
        type: Action.ADD_IMAGES_DATA,
        payload: {
            imageData,
        },
    };
}

export function updateImageData(imageData: ImageData[]): LabelsActionTypes {
    return {
        type: Action.UPDATE_IMAGES_DATA,
        payload: {
            imageData,
        },
    };
}

export function updateLabelNames(labels: LabelName[]): LabelsActionTypes {
    return {
        type: Action.UPDATE_LABEL_NAMES,
        payload: {
            labels
        }
    }
}

export function updateFirstLabelCreatedFlag(firstLabelCreatedFlag: boolean): LabelsActionTypes {
    return {
        type: Action.UPDATE_FIRST_LABEL_CREATED_FLAG,
        payload: {
            firstLabelCreatedFlag
        }
    }
}

export function updateSubjects(subjects: Subject[]): LabelsActionTypes {
    return {
        type: Action.UPDATE_SUBJECTS,
        payload: {
            subjects
        }
    }
}

export function updateActiveSubjectId(activeSubjectId: string | null): LabelsActionTypes {
    return {
        type: Action.UPDATE_ACTIVE_SUBJECT_ID,
        payload: {
            activeSubjectId
        }
    }
}

// Focus mode: when set, the editor narrows the timeline + behavior bar
// to the picked behavior. Pass `null` to exit focus.
export function updateFocusedLabelNameId(focusedLabelNameId: string | null): LabelsActionTypes {
    return {
        type: Action.UPDATE_FOCUSED_LABEL_NAME_ID,
        payload: {
            focusedLabelNameId
        }
    }
}
