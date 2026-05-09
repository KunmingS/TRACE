import {HotKeyAction} from "../../data/HotKeyAction";
import {BaseContext} from "./BaseContext";
import {ImageActions} from "../actions/ImageActions";
import {ViewPortActions} from "../actions/ViewPortActions";
import {PlatformUtil} from "../../utils/PlatformUtil";
import {LabelActions} from "../actions/LabelActions";

export class EditorContext extends BaseContext {
    public static actions: HotKeyAction[] = [
        {
            keyCombo: PlatformUtil.isMac(window.navigator.userAgent) ? ["Meta", "ArrowLeft"] : ["Control", "ArrowLeft"],
            action: (event: KeyboardEvent) => {
                ImageActions.getPreviousImage()
            }
        },
        {
            keyCombo: PlatformUtil.isMac(window.navigator.userAgent) ? ["Meta", "ArrowRight"] : ["Control", "ArrowRight"],
            action: (event: KeyboardEvent) => {
                ImageActions.getNextImage();
            }
        },
        {
            keyCombo: PlatformUtil.isMac(window.navigator.userAgent) ? ["Meta", "="] : ["Control", "="],
            action: (event: KeyboardEvent) => {
                ViewPortActions.zoomIn();
            }
        },
        {
            keyCombo: PlatformUtil.isMac(window.navigator.userAgent) ? ["Meta", "+"] : ["Control", "+"],
            action: (event: KeyboardEvent) => {
                ViewPortActions.zoomIn();
            }
        },
        {
            keyCombo: PlatformUtil.isMac(window.navigator.userAgent) ? ["Meta", "-"] : ["Control", "-"],
            action: (event: KeyboardEvent) => {
                ViewPortActions.zoomOut();
            }
        },
        {
            keyCombo: ["Backspace"],
            action: (event: KeyboardEvent) => {
                LabelActions.deleteActiveLabel();
            }
        },
        {
            keyCombo: ["Delete"],
            action: (event: KeyboardEvent) => {
                LabelActions.deleteActiveLabel();
            }
        },
    ];
}
