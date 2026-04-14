import {LabelType} from '../data/enums/LabelType';
import {ProjectType} from '../data/enums/ProjectType';

export interface ILabelToolkit {
    labelType: LabelType;
    headerText: string;
    imageSrc: string;
    imageAlt: string;
    projectType: ProjectType;
} 