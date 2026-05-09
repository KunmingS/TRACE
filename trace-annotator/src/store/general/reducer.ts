import {GeneralActionTypes, GeneralState} from './types';
import {Action} from '../Actions';
import {CustomCursorStyle} from '../../data/enums/CustomCursorStyle';
import {ViewPointSettings} from '../../settings/ViewPointSettings';

const initialState: GeneralState = {
    windowSize: null,
    activePopupType: null,
    popupPayload: null,
    customCursorStyle: CustomCursorStyle.DEFAULT,
    activeContext: null,
    preventCustomCursor: false,
    imageDragMode: false,
    crossHairVisible: true,
    enablePerClassColoration: true,
    projectData: {
        type: null,
        name: 'my-project-name',
    },
    zoom: ViewPointSettings.MIN_ZOOM,
    jumpToFrameIndex: null,
    videoDirectory: '',
    videoFiles: [],
    videoCsvOverrides: {},
    homeTab: 'annotate'
};

export function generalReducer(
    state = initialState,
    action: GeneralActionTypes
): GeneralState {
    switch (action.type) {
        case Action.UPDATE_WINDOW_SIZE: {
            return {
                ...state,
                windowSize: action.payload.windowSize
            }
        }
        case Action.UPDATE_ACTIVE_POPUP_TYPE: {
            // Closing a popup also clears any payload it was carrying so a
            // future popup doesn't see stale state.
            const next = action.payload.activePopupType;
            return {
                ...state,
                activePopupType: next,
                popupPayload: next ? state.popupPayload : null
            }
        }
        case Action.UPDATE_POPUP_PAYLOAD: {
            return {
                ...state,
                popupPayload: action.payload.popupPayload
            }
        }
        case Action.UPDATE_CUSTOM_CURSOR_STYLE: {
            return {
                ...state,
                customCursorStyle: action.payload.customCursorStyle
            }
        }
        case Action.UPDATE_CONTEXT: {
            return {
                ...state,
                activeContext: action.payload.activeContext
            }
        }
        case Action.UPDATE_PREVENT_CUSTOM_CURSOR_STATUS: {
            return {
                ...state,
                preventCustomCursor: action.payload.preventCustomCursor
            }
        }
        case Action.UPDATE_IMAGE_DRAG_MODE_STATUS: {
            return {
                ...state,
                imageDragMode: action.payload.imageDragMode
            }
        }
        case Action.UPDATE_CROSS_HAIR_VISIBLE_STATUS: {
            return {
                ...state,
                crossHairVisible: action.payload.crossHairVisible
            }
        }
        case Action.UPDATE_PROJECT_DATA: {
            return {
                ...state,
                projectData: action.payload.projectData
            }
        }
        case Action.UPDATE_ZOOM: {
            return {
                ...state,
                zoom: action.payload.zoom
            }
        }
        case Action.UPDATE_ENABLE_PER_CLASS_COLORATION_STATUS: {
            return {
                ...state,
                enablePerClassColoration: action.payload.enablePerClassColoration
            }
        }
        case Action.JUMP_TO_FRAME: {
            return {
                ...state,
                jumpToFrameIndex: action.payload.frameIndex
            };
        }
        case Action.UPDATE_VIDEO_DIRECTORY: {
            const next = action.payload.videoDirectory;
            // Filename-keyed overrides only make sense for the current
            // directory, so a directory change invalidates them. Keep them
            // when the dir hasn't actually changed (e.g. a same-dir reload).
            const overrides = next === state.videoDirectory
                ? state.videoCsvOverrides
                : {};
            return {
                ...state,
                videoDirectory: next,
                videoCsvOverrides: overrides
            };
        }
        case Action.UPDATE_VIDEO_FILES: {
            return {
                ...state,
                videoFiles: action.payload.videoFiles
            };
        }
        case Action.MARK_VIDEO_HAS_CSV: {
            const { filename } = action.payload;
            if (state.videoCsvOverrides[filename]) return state;
            return {
                ...state,
                videoCsvOverrides: {
                    ...state.videoCsvOverrides,
                    [filename]: true
                }
            };
        }
        case Action.CLEAR_VIDEO_HAS_CSV: {
            const { filename } = action.payload;
            if (!state.videoCsvOverrides[filename]) return state;
            const videoCsvOverrides = { ...state.videoCsvOverrides };
            delete videoCsvOverrides[filename];
            return {
                ...state,
                videoCsvOverrides
            };
        }
        case Action.UPDATE_HOME_TAB: {
            return {
                ...state,
                homeTab: action.payload.homeTab
            };
        }
        default:
            return state;
    }
}
