import {IRect} from '../interfaces/IRect';
import {ISize} from '../interfaces/ISize';
import {IPoint} from '../interfaces/IPoint';

export type EditorData = {
    mousePositionOnViewPortContent: IPoint;
    viewPortContentSize: ISize;
    viewPortContentImageRect: IRect;
    realImageSize: ISize;
    absoluteViewPortContentScrollPosition: IPoint;
    event?: Event;
}
