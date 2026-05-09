import {ImageData, LabelName, Subject} from '../../store/labels/types';
import {LabelType} from '../../data/enums/LabelType';

export type ImportResult = {
    imagesData: ImageData[]
    labelNames: LabelName[]
    subjects?: Subject[]
}

export type ImportOnSuccess = (
    imagesData: ImageData[],
    labelNames: LabelName[],
    subjects?: Subject[]
) => any;

export class AnnotationImporter {
    public labelType: LabelType[]

    constructor(labelType: LabelType[]) {
        this.labelType = labelType;
    }

    public import(
        filesData: File[],
        onSuccess: ImportOnSuccess,
        onFailure: (error?:Error) => any
    ): void {
        throw new Error('Method not implemented.');
    }
}