import {store} from '../..';
import {PopupWindowType} from '../../data/enums/PopupWindowType';
import {ContextType} from '../../data/enums/ContextType';
import {CustomCursorStyle} from '../../data/enums/CustomCursorStyle';
import {ProjectType} from '../../data/enums/ProjectType';
import {ISize} from '../../interfaces/ISize';

export class GeneralSelector {
    public static getActivePopupType(): PopupWindowType {
        return store.getState().general.activePopupType;
    }

    public static getActiveContext(): ContextType {
        return store.getState().general.activeContext;
    }

    public static getPreventCustomCursorStatus(): boolean {
        return store.getState().general.preventCustomCursor;
    }

    public static getImageDragModeStatus(): boolean {
        return store.getState().general.imageDragMode;
    }

    public static getCrossHairVisibleStatus(): boolean {
        return store.getState().general.crossHairVisible;
    }

    public static getCustomCursorStyle(): CustomCursorStyle {
        return store.getState().general.customCursorStyle;
    }

    public static getProjectName(): string {
        return store.getState().general.projectData.name;
    }

    public static getProjectType(): ProjectType {
        return store.getState().general.projectData.type;
    }

    public static getZoom(): number {
        return store.getState().general.zoom;
    }

    public static getEnablePerClassColorationStatus(): boolean {
        return store.getState().general.enablePerClassColoration;
    }

    public static getWindowSize(): ISize {
        return store.getState().general.windowSize;
    }

    public static isObjectDetectorLoaded(): boolean {
        return store.getState().ai.isSSDObjectDetectorLoaded;
    }

    public static isPoseDetectionLoaded(): boolean {
        return store.getState().ai.isPoseDetectorLoaded;
    }

    public static isYOLOV5ObjectDetectorLoaded(): boolean {
        return store.getState().ai.isYOLOV5ObjectDetectorLoaded;
    }

    public static getRoboflowAPIDetails(): {
        model: string;
        key: string;
        status: boolean;
    } {
        return store.getState().ai.roboflowAPIDetails;
    }
}
