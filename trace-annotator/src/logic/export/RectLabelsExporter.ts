import {AnnotationFormatType} from '../../data/enums/AnnotationFormatType';
import {ImageData, LabelName, LabelRect, Subject} from '../../store/labels/types';
import {LabelsSelector} from '../../store/selectors/LabelsSelector';
import {ExporterUtil} from '../../utils/ExporterUtil';
import {Settings} from '../../settings/Settings';
import {DEFAULT_SUBJECT_ID} from '../../store/labels/reducer';

// Build the `# trace-meta:` line carrying behavior→shortcut bindings so the
// CSV round-trips shortcut bindings across save/load. URL-encoding defends
// against names that might contain commas or colons.
export function buildTraceMetaLine(labelNames: LabelName[]): string | null {
    const pairs = labelNames
        .filter(ln => !!ln.shortcut)
        .map(ln => `${encodeURIComponent(ln.name)}:${encodeURIComponent(ln.shortcut!)}`);
    if (pairs.length === 0) return null;
    return `# trace-meta: behaviors=${pairs.join(',')}`;
}

export class RectLabelsExporter {
    public static export(exportFormatType: AnnotationFormatType): void {
        if (exportFormatType === AnnotationFormatType.CSV) {
            RectLabelsExporter.exportAsCSV();
        }
    }

    public static wrapRectLabelIntoCSV(
        labelRect: LabelRect,
        imageName: string,
        subjectNameById: Map<string, string>,
        fallbackSubjectName: string
    ): string {
        const animalId = labelRect.animalId ?? DEFAULT_SUBJECT_ID;
        const animalName = subjectNameById.get(animalId) ?? fallbackSubjectName;
        const labelFields = [
            imageName,
            labelRect.timestamp.toString(),
            labelRect.endTimestamp.toString(),
            animalName,
            labelRect.behavior || '',
        ];
        return labelFields.join(Settings.CSV_SEPARATOR)
    }

    private static exportAsCSV(): void {
        try {
            const subjects: Subject[] = LabelsSelector.getSubjects();
            const subjectNameById = new Map(subjects.map(s => [s.id, s.name]));
            const fallbackSubjectName = subjects[0]?.name ?? 'Animal 1';
            const contentEntries: string[] = LabelsSelector.getImagesData()
                .map((imageData: ImageData) => {
                    return RectLabelsExporter.wrapRectLabelsIntoCSV(
                        imageData, subjectNameById, fallbackSubjectName)})
                .filter((imageLabelData: string) => {
                    return !!imageLabelData})
            contentEntries.unshift(Settings.RECT_LABELS_EXPORT_CSV_COLUMN_NAMES)

            // Persist behavior→shortcut bindings as the very first line.
            // Older readers (before the trace-meta extension) tolerate this
            // because comment lines beginning with `#` are skipped by the
            // CSVImporter; pure pandas readers will fail unless given the
            // `comment='#'` option, which mirrors common conventions.
            const metaLine = buildTraceMetaLine(LabelsSelector.getLabelNames());
            if (metaLine) contentEntries.unshift(metaLine);

            const content: string = contentEntries.join('\n');
            const fileName: string = `${ExporterUtil.getExportFileName()}.csv`;
            ExporterUtil.saveAs(content, fileName);
        } catch (error) {
            throw new Error(error as string);
        }
    }

    private static wrapRectLabelsIntoCSV(
        imageData: ImageData,
        subjectNameById: Map<string, string>,
        fallbackSubjectName: string
    ): string {
        if (imageData.labelRects.length === 0)
            return null;

        const labelRectsString: string[] = imageData.labelRects
            .filter((labelRect: LabelRect) => labelRect.labelId !== null)
            .map((labelRect: LabelRect) => RectLabelsExporter.wrapRectLabelIntoCSV(
                labelRect, imageData.fileData.name, subjectNameById, fallbackSubjectName));
        return labelRectsString.join('\n');
    }
}
