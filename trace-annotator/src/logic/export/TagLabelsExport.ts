import {AnnotationFormatType} from "../../data/enums/AnnotationFormatType";
import {LabelsSelector} from "../../store/selectors/LabelsSelector";
import {ImageData, LabelName} from "../../store/labels/types";
import {ExporterUtil} from "../../utils/ExporterUtil";
import {findLast} from "lodash";

export class TagLabelsExporter {
    public static export(exportFormatType: AnnotationFormatType): void {
        if (exportFormatType === AnnotationFormatType.CSV) {
            TagLabelsExporter.exportAsCSV();
        }
    }

    private static exportAsCSV(): void {
        const content: string = LabelsSelector.getImagesData()
            .filter((imageData: ImageData) => {
                return imageData.labelNameIds.length > 0
            })
            .map((imageData: ImageData) => {
                return TagLabelsExporter.wrapLabelNamesIntoCSV(imageData)})
            .join("\n");
        const fileName: string = `${ExporterUtil.getExportFileName()}.csv`;
        ExporterUtil.saveAs(content, fileName);
    }

    private static wrapLabelNamesIntoCSV(imageData: ImageData): string {
        if (imageData.labelNameIds.length === 0 || !imageData.loadStatus)
            return null;

        const labelNames: LabelName[] = LabelsSelector.getLabelNames();
        const annotations: string[] = imageData.labelNameIds.map((labelNameId: string) => {
            return findLast(labelNames, {id: labelNameId}).name;
        })
        const labelFields = annotations.length !== 0 ? [
            imageData.fileData.name,
            `"[${annotations.toString()}]"`
        ] : [];
        return labelFields.join(",")
    }
}
