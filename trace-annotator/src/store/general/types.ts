import {ISize} from '../../interfaces/ISize';
import {Action} from '../Actions';
import {PopupWindowType} from '../../data/enums/PopupWindowType';
import {CustomCursorStyle} from '../../data/enums/CustomCursorStyle';
import {ContextType} from '../../data/enums/ContextType';
import {ProjectType} from '../../data/enums/ProjectType';

export type ProjectData = {
    type: ProjectType;
    name: string,
}

export type HomeTab = 'annotate' | 'pipeline' | 'tutorial';

export type GeneralState = {
    windowSize: ISize;
    activePopupType: PopupWindowType;
    customCursorStyle: CustomCursorStyle;
    preventCustomCursor: boolean;
    imageDragMode: boolean;
    crossHairVisible: boolean;
    enablePerClassColoration: boolean;
    activeContext: ContextType;
    projectData: ProjectData;
    zoom: number;
    jumpToFrameIndex: number | null;
    videoDirectory: string;
    videoFiles: string[];
    homeTab: HomeTab;
}

interface UpdateProjectData {
    type: typeof Action.UPDATE_PROJECT_DATA;
    payload: {
        projectData: ProjectData;
    }
}

interface UpdateWindowSize {
    type: typeof Action.UPDATE_WINDOW_SIZE;
    payload: {
        windowSize: ISize;
    }
}

interface UpdateActivePopupType {
    type: typeof Action.UPDATE_ACTIVE_POPUP_TYPE;
    payload: {
        activePopupType: PopupWindowType;
    }
}

interface UpdateCustomCursorStyle {
    type: typeof Action.UPDATE_CUSTOM_CURSOR_STYLE;
    payload: {
        customCursorStyle: CustomCursorStyle;
    }
}

interface UpdateActiveContext {
    type: typeof Action.UPDATE_CONTEXT;
    payload: {
        activeContext: ContextType;
    }
}

interface UpdatePreventCustomCursorStatus {
    type: typeof Action.UPDATE_PREVENT_CUSTOM_CURSOR_STATUS;
    payload: {
        preventCustomCursor: boolean;
    }
}

interface UpdateImageDragModeStatus {
    type: typeof Action.UPDATE_IMAGE_DRAG_MODE_STATUS;
    payload: {
        imageDragMode: boolean;
    }
}

interface UpdateCrossHairVisibleStatus {
    type: typeof Action.UPDATE_CROSS_HAIR_VISIBLE_STATUS;
    payload: {
        crossHairVisible: boolean;
    }
}

interface UpdateZoom {
    type: typeof Action.UPDATE_ZOOM,
    payload: {
        zoom: number;
    }
}

interface UpdatePerClassColoration {
    type: typeof Action.UPDATE_ENABLE_PER_CLASS_COLORATION_STATUS,
    payload: {
        enablePerClassColoration: boolean;
    }
}

interface JumpToFrame {
    type: typeof Action.JUMP_TO_FRAME;
    payload: {
        frameIndex: number | null;
    }
}

interface UpdateVideoDirectory {
    type: typeof Action.UPDATE_VIDEO_DIRECTORY;
    payload: {
        videoDirectory: string;
    }
}

interface UpdateVideoFiles {
    type: typeof Action.UPDATE_VIDEO_FILES;
    payload: {
        videoFiles: string[];
    }
}

interface UpdateHomeTab {
    type: typeof Action.UPDATE_HOME_TAB;
    payload: {
        homeTab: HomeTab;
    }
}

export type GeneralActionTypes = UpdateProjectData
    | UpdateWindowSize
    | UpdateActivePopupType
    | UpdateCustomCursorStyle
    | UpdateActiveContext
    | UpdatePreventCustomCursorStatus
    | UpdateImageDragModeStatus
    | UpdateCrossHairVisibleStatus
    | UpdateZoom
    | UpdatePerClassColoration
    | JumpToFrame
    | UpdateVideoDirectory
    | UpdateVideoFiles
    | UpdateHomeTab
