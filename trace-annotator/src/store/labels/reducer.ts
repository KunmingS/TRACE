import {LabelsActionTypes, LabelsState, ImageData} from './types';
import {Action} from '../Actions';

export const DEFAULT_SUBJECT_ID = 'animal_0';

const initialState: LabelsState = {
    activeImageIndex: null,
    activeLabelNameId: null,
    activeLabelType: null,
    activeLabelId: null,
    highlightedLabelId: null,
    imagesData: [],
    firstLabelCreatedFlag: false,
    labels: [],
    subjects: [{ id: DEFAULT_SUBJECT_ID, name: 'Animal 1' }],
    activeSubjectId: DEFAULT_SUBJECT_ID,
    focusedLabelNameId: null
};

export function labelsReducer(
    state = initialState,
    action: LabelsActionTypes
): LabelsState {
    switch (action.type) {
        case Action.UPDATE_ACTIVE_IMAGE_INDEX: {
            return {
                ...state,
                activeImageIndex: action.payload.activeImageIndex
            }
        }
        case Action.UPDATE_ACTIVE_LABEL_NAME_ID: {
            return {
                ...state,
                activeLabelNameId: action.payload.activeLabelNameId
            }
        }
        case Action.UPDATE_ACTIVE_LABEL_ID: {
            return {
                ...state,
                activeLabelId: action.payload.activeLabelId
            }
        }
        case Action.UPDATE_HIGHLIGHTED_LABEL_ID: {
            return {
                ...state,
                highlightedLabelId: action.payload.highlightedLabelId
            }
        }
        case Action.UPDATE_ACTIVE_LABEL_TYPE: {
            return {
                ...state,
                activeLabelType: action.payload.activeLabelType
            }
        }
        case Action.UPDATE_IMAGE_DATA_BY_ID: {
            return {
                ...state,
                imagesData: state.imagesData.map((imageData: ImageData) =>
                    imageData.id === action.payload.id ? action.payload.newImageData : imageData
                )
            }
        }
        case Action.ADD_IMAGES_DATA: {
            return {
                ...state,
                imagesData: state.imagesData.concat(action.payload.imageData)
            }
        }
        case Action.UPDATE_IMAGES_DATA: {
            return {
                ...state,
                imagesData: action.payload.imageData
            }
        }
        case Action.UPDATE_LABEL_NAMES: {
            return {
                ...state,
                labels: action.payload.labels
            }
        }
        case Action.UPDATE_FIRST_LABEL_CREATED_FLAG: {
            return {
                ...state,
                firstLabelCreatedFlag: action.payload.firstLabelCreatedFlag
            }
        }
        case Action.UPDATE_SUBJECTS: {
            const subjects = action.payload.subjects;
            const stillExists = subjects.some(s => s.id === state.activeSubjectId);
            const nextActive = stillExists
                ? state.activeSubjectId
                : (subjects[0]?.id ?? null);
            return {
                ...state,
                subjects,
                activeSubjectId: nextActive
            }
        }
        case Action.UPDATE_ACTIVE_SUBJECT_ID: {
            return {
                ...state,
                activeSubjectId: action.payload.activeSubjectId
            }
        }
        case Action.UPDATE_FOCUSED_LABEL_NAME_ID: {
            return {
                ...state,
                focusedLabelNameId: action.payload.focusedLabelNameId
            }
        }
        default:
            return state;
    }
}
