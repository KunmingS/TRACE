import {ILabelToolkit} from '../../interfaces/ILabelToolkit';
import {LabelType} from '../enums/LabelType';
import {ProjectType} from '../enums/ProjectType';

export const LabelToolkitData: ILabelToolkit[] = [
    {
        labelType: LabelType.RECT,
        headerText: 'Rect',
        imageSrc: 'ico/rectangle.png',
        imageAlt: 'rect',
        projectType: ProjectType.IMAGE
    },
    {
        labelType: LabelType.VIDEO_RECOGNITION,
        headerText: 'Video Recognition',
        imageSrc: 'ico/tag.png',
        imageAlt: 'video-recognition',
        projectType: ProjectType.VIDEO
    }
];
