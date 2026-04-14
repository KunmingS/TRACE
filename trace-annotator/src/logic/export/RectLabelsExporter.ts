import {AnnotationFormatType} from '../../data/enums/AnnotationFormatType';
import {ImageData, LabelRect} from '../../store/labels/types';
import {LabelsSelector} from '../../store/selectors/LabelsSelector';
import {ExporterUtil} from '../../utils/ExporterUtil';
import {Settings} from '../../settings/Settings';

export class RectLabelsExporter {
    public static export(exportFormatType: AnnotationFormatType): void {
        if (exportFormatType === AnnotationFormatType.CSV) {
            RectLabelsExporter.exportAsCSV();
        }
    }

    public static wrapRectLabelIntoCSV(
        labelRect: LabelRect,
        imageName: string
    ): string {
        const labelFields = [
            imageName,
            labelRect.timestamp.toString(),
            labelRect.endTimestamp.toString(),
            labelRect.behavior || '',
        ];
        return labelFields.join(Settings.CSV_SEPARATOR)
    }

    private static exportAsCSV(): void {
        try {
            const contentEntries: string[] = LabelsSelector.getImagesData()
                .map((imageData: ImageData) => {
                    return RectLabelsExporter.wrapRectLabelsIntoCSV(imageData)})
                .filter((imageLabelData: string) => {
                    return !!imageLabelData})
            contentEntries.unshift(Settings.RECT_LABELS_EXPORT_CSV_COLUMN_NAMES)

            const content: string = contentEntries.join('\n');
            const fileName: string = `${ExporterUtil.getExportFileName()}.csv`;
            ExporterUtil.saveAs(content, fileName);
        } catch (error) {
            throw new Error(error as string);
        }
    }

    private static wrapRectLabelsIntoCSV(imageData: ImageData): string {
        if (imageData.labelRects.length === 0)
            return null;

        const labelRectsString: string[] = imageData.labelRects
            .filter((labelRect: LabelRect) => labelRect.labelId !== null)
            .map((labelRect: LabelRect) => RectLabelsExporter.wrapRectLabelIntoCSV(
                labelRect, imageData.fileData.name));
        return labelRectsString.join('\n');
    }
}
