import {Action} from '../Actions';
import {LabelType} from '../../data/enums/LabelType';
import {LabelStatus} from '../../data/enums/LabelStatus';

export type Annotation = {
    id: string;
    labelId: string | null;
    isVisible: boolean;
}

export type LabelRect = Annotation & {
    isCreatedByAI: boolean;
    status: LabelStatus;
    suggestedLabel: string;
    timestamp?: string;
    frame?: number;
    endTimestamp?: string;
    endFrame?: number;
    behavior?: string;
    animalId?: string;
}

export type LabelName = {
    name: string;
    id: string;
    color?: string;
    shortcut?: string;
}

export type Subject = {
    id: string;
    name: string;
}

export type VideoData = {
    id: string;
    fileData: File;
    loadStatus: boolean;
    duration: number;
    fps: number;
    width: number;
    height: number;
    frames: VideoFrame[];
    currentFrameIndex: number;
}

export type VideoFrame = {
    id: string;
    frameIndex: number;
    timestamp: number;
    imageData: ImageData;
    isKeyFrame: boolean;
}

export type ImageData = {
    id: string;
    fileData: File;
    loadStatus: boolean;
    labelRects: LabelRect[];
    labelNameIds: string[];
    frameIndex?: number;
    timestamp?: number;
    frameRate?: number;
    videoUrl?: string;
}

export type LabelsState = {
    activeImageIndex: number;
    activeLabelNameId: string;
    activeLabelType: LabelType;
    activeLabelId: string | null;
    highlightedLabelId: string;
    imagesData: ImageData[];
    firstLabelCreatedFlag: boolean;
    labels: LabelName[];
    subjects: Subject[];
    activeSubjectId: string | null;
    // When non-null, the editor is "focused" on a single behavior:
    // the timeline hides clips of every other behavior, the top bar
    // hides every other chip, and Shift/Cmd+Arrow boundary navigation
    // is filtered to the focused behavior's clips. Click the focused
    // chip again to exit.
    focusedLabelNameId: string | null;
}

interface UpdateActiveImageIndex {
    type: typeof Action.UPDATE_ACTIVE_IMAGE_INDEX;
    payload: {
        activeImageIndex: number;
    }
}

interface UpdateActiveLabelNameId {
    type: typeof Action.UPDATE_ACTIVE_LABEL_NAME_ID;
    payload: {
        activeLabelNameId: string;
    }
}

interface UpdateActiveLabelId {
    type: typeof Action.UPDATE_ACTIVE_LABEL_ID;
    payload: {
        activeLabelId: string;
    }
}

interface UpdateHighlightedLabelId {
    type: typeof Action.UPDATE_HIGHLIGHTED_LABEL_ID;
    payload: {
        highlightedLabelId: string;
    }
}

interface UpdateActiveLabelType {
    type: typeof Action.UPDATE_ACTIVE_LABEL_TYPE;
    payload: {
        activeLabelType: LabelType;
    }
}

interface UpdateImageDataById {
    type: typeof Action.UPDATE_IMAGE_DATA_BY_ID;
    payload: {
        id: string;
        newImageData: ImageData;
    }
}

interface AddImageData {
    type: typeof Action.ADD_IMAGES_DATA;
    payload: {
        imageData: ImageData[];
    }
}

interface UpdateImageData {
    type: typeof Action.UPDATE_IMAGES_DATA;
    payload: {
        imageData: ImageData[];
    }
}

interface UpdateLabelNames {
    type: typeof Action.UPDATE_LABEL_NAMES;
    payload: {
        labels: LabelName[];
    }
}

interface UpdateFirstLabelCreatedFlag {
    type: typeof Action.UPDATE_FIRST_LABEL_CREATED_FLAG;
    payload: {
        firstLabelCreatedFlag: boolean;
    }
}

interface UpdateSubjects {
    type: typeof Action.UPDATE_SUBJECTS;
    payload: {
        subjects: Subject[];
    }
}

interface UpdateActiveSubjectId {
    type: typeof Action.UPDATE_ACTIVE_SUBJECT_ID;
    payload: {
        activeSubjectId: string | null;
    }
}

interface UpdateFocusedLabelNameId {
    type: typeof Action.UPDATE_FOCUSED_LABEL_NAME_ID;
    payload: {
        focusedLabelNameId: string | null;
    }
}

export type LabelsActionTypes = UpdateActiveImageIndex
    | UpdateActiveLabelNameId
    | UpdateActiveLabelType
    | UpdateImageDataById
    | AddImageData
    | UpdateImageData
    | UpdateLabelNames
    | UpdateActiveLabelId
    | UpdateHighlightedLabelId
    | UpdateFirstLabelCreatedFlag
    | UpdateSubjects
    | UpdateActiveSubjectId
    | UpdateFocusedLabelNameId
