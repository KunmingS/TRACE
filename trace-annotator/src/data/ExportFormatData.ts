import {LabelType} from './enums/LabelType';
import {AnnotationFormatType} from './enums/AnnotationFormatType';
import {ILabelFormatData} from '../interfaces/ILabelFormatData';

export const ExportFormatData: Record<LabelType, ILabelFormatData[]> = {
    [LabelType.RECT]: [
        {
            type: AnnotationFormatType.CSV,
            label: 'Single CSV file'
        }
    ],
    [LabelType.VIDEO_RECOGNITION]: [
        {
            type: AnnotationFormatType.CSV,
            label: 'Single CSV file'
        }
    ]
};
