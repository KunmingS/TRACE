import {EditorModel} from '../../staticModels/EditorModel';
import {NumberUtil} from '../../utils/NumberUtil';
import {ViewPointSettings} from '../../settings/ViewPointSettings';
import {ISize} from '../../interfaces/ISize';
import {IPoint} from '../../interfaces/IPoint';
import {GeneralSelector} from '../../store/selectors/GeneralSelector';
import {store} from '../../index';
import {updateZoom} from '../../store/general/actionCreators';

export class ViewPortActions {
    public static updateViewPortSize() {
        if (!!EditorModel.editor) {
            EditorModel.viewPortSize = {
                width: EditorModel.editor.offsetWidth,
                height: EditorModel.editor.offsetHeight
            }
        }
    }

    public static calculateViewPortContentSize(): ISize {
        return null;
    }

    public static calculateViewPortContentImageRect() {
        return null;
    }

    public static resizeViewPortContent() {
        // No-op: video editor doesn't use canvas
    }

    public static getRelativeScrollPosition(): IPoint {
        return null;
    }

    public static getAbsoluteScrollPosition(): IPoint {
        return null;
    }

    public static setScrollPosition(position: IPoint) {
        // No-op: video editor doesn't use scrollbars
    }

    public static translateViewPortPosition() {
        // No-op
    }

    public static zoomIn() {
        if (EditorModel.viewPortActionsDisabled) return;
        const currentZoom: number = GeneralSelector.getZoom();
        ViewPortActions.setZoom(currentZoom + ViewPointSettings.ZOOM_STEP);
    }

    public static zoomOut() {
        if (EditorModel.viewPortActionsDisabled) return;
        const currentZoom: number = GeneralSelector.getZoom();
        ViewPortActions.setZoom(currentZoom - ViewPointSettings.ZOOM_STEP);
    }

    public static setDefaultZoom() {
        ViewPortActions.setZoom(ViewPointSettings.MIN_ZOOM);
    }

    public static setOneForOneZoom() {
        // No-op without image reference
    }

    public static setZoom(value: number) {
        const currentZoom: number = GeneralSelector.getZoom();
        const isNewValueValid: boolean = NumberUtil.isValueInRange(
            value, ViewPointSettings.MIN_ZOOM, ViewPointSettings.MAX_ZOOM);
        if (isNewValueValid && value !== currentZoom) {
            store.dispatch(updateZoom(value));
        }
    }
}
