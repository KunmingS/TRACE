import { IRect } from "../interfaces/IRect";
import { IPoint } from "../interfaces/IPoint";
import { ISize } from "../interfaces/ISize";

export class EditorModel {
    public static editor: HTMLDivElement;
    public static image: HTMLImageElement;

    public static isLoading: boolean = false;
    public static viewPortActionsDisabled: boolean = false;
    public static mousePositionOnViewPortContent: IPoint;
    public static viewPortSize: ISize;
    public static defaultRenderImageRect: IRect;
}
