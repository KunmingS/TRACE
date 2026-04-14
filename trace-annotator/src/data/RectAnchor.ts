import {Direction} from './enums/Direction';
import {IPoint} from '../interfaces/IPoint';

export type RectAnchor = {
    type: Direction;
    position: IPoint;
}
