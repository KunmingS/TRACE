import {AnnotationFormatType} from './enums/AnnotationFormatType';
import {CSVImporter} from '../logic/import/csv/CSVImporter';
import {AnnotationImporter} from '../logic/import/AnnotationImporter';

type ImporterSpecDataMap = {
    [key in AnnotationFormatType]: typeof AnnotationImporter;
}

export const ImporterSpecData: ImporterSpecDataMap = {
    [AnnotationFormatType.CSV]: CSVImporter as unknown as typeof AnnotationImporter
};
