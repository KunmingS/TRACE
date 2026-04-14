import {EditorModel} from "../../staticModels/EditorModel";

export class EditorActions {
    public static setLoadingStatus(status: boolean) {
        EditorModel.isLoading = status;
    }

    public static setActiveImage(image: HTMLImageElement) {
        EditorModel.image = image;
    }

    public static setViewPortActionsDisabledStatus(status: boolean) {
        EditorModel.viewPortActionsDisabled = status;
    }

    public static fullRender() {
        // No-op: video editor uses Plyr, not canvas rendering
    }
}
