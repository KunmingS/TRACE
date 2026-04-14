import {ImageData, LabelName, LabelRect} from '../../../store/labels/types';
import {LabelStatus} from '../../../data/enums/LabelStatus';
import {LabelsSelector} from '../../../store/selectors/LabelsSelector';
import { v4 as uuidv4 } from 'uuid';
import {ArrayUtil, PartitionResult} from '../../../utils/ArrayUtil';
import {ImageDataUtil} from '../../../utils/ImageDataUtil';
import {LabelUtil} from '../../../utils/LabelUtil';
import {
    CSVAnnotationDeserializationError,
    CSVAnnotationFileCountError,
    CSVAnnotationReadingError,
    CSVFormatValidationError
} from './CSVErrors';
import {LabelType} from '../../../data/enums/LabelType';
import {AnnotationImporter, ImportResult} from '../AnnotationImporter';
import {Settings} from "../../../settings/Settings";

export type CSVRow = {
    // imageName: string;
    // For frame-based CSV
    startFrame?: number;
    endFrame?: number;
    // For time-based CSV
    timestamp?: number;
    endTimestamp?: number;
    // label name or behavior
    behavior: string;
}

export class CSVImporter extends AnnotationImporter {
    public static requiredColumns = ['start_frame', 'end_frame', 'behavior']

    public import(
        filesData: File[],
        onSuccess: (imagesData: ImageData[], labelNames: LabelName[]) => any,
        onFailure: (error?:Error) => any
    ): void {
        if (filesData.length > 1) {
            onFailure(new CSVAnnotationFileCountError());
        }

        const reader = new FileReader();
        reader.readAsText(filesData[0]);
        reader.onloadend = (evt: any) => {
            try {
                const inputImagesData: ImageData[] = LabelsSelector.getImagesData();
                const csvRows = CSVImporter.parseCSV(evt.target.result);
                const {imagesData, labelNames} = this.applyLabels(inputImagesData, csvRows);
                onSuccess(imagesData, labelNames);
            } catch (error) {
                onFailure(error as Error);
            }
        };
        reader.onerror = () => onFailure(new CSVAnnotationReadingError());
    }

    public static parseCSV(text: string): CSVRow[] {
        try {
            const lines = text.trim().split('\n');
            if (lines.length < 2) {
                throw new CSVFormatValidationError('CSV file must contain at least a header and one data row');
            }

            const header = lines[0].split(Settings.CSV_SEPARATOR).map(col => col.trim());
            // Detect format: either frame-based or time-based
            const lower = header.map(h => h.toLowerCase());
            const hasFrameCols = ['start_frame','end_frame','behavior'].every(c => lower.includes(c));
            const hasTimeCols = ['labelid','timestamp','endtimestamp'].every(c => lower.includes(c));
            if (!hasFrameCols && !hasTimeCols) {
                throw new CSVFormatValidationError(`CSV missing required columns; expected start_frame, end_frame, behavior OR labelId, timestamp, endTimestamp`);
            }

            const rows: CSVRow[] = [];
            for (let i = 1; i < lines.length; i++) {
                const values = lines[i].split(Settings.CSV_SEPARATOR).map(val => val.trim());
                if (values.length !== header.length) {
                    continue; // Skip malformed rows
                }

                // Extract image name from the first column if it exists
                // const imageNameIndex = header.indexOf('image_name');
                let row: CSVRow;
                if (hasTimeCols) {
                    // time-based format: labelId|behavior, timestamp, endTimestamp
                    const idxLabel = lower.indexOf('labelid');
                    const idxTs = lower.indexOf('timestamp');
                    const idxEnd = lower.indexOf('endtimestamp');
                    row = {
                        behavior: values[idxLabel],
                        timestamp: parseFloat(values[idxTs] || '0'),
                        endTimestamp: parseFloat(values[idxEnd] || '0')
                    };
                } else {
                    // frame-based format
                    const idxStart = lower.indexOf('start_frame');
                    const idxEndF = lower.indexOf('end_frame');
                    const idxBeh = lower.indexOf('behavior');
                    row = {
                        startFrame: parseInt(values[idxStart] || '0', 10),
                        endFrame: parseInt(values[idxEndF] || '0', 10),
                        behavior: values[idxBeh] || ''
                    };
                }
                rows.push(row);
            }

            return rows;
        } catch (error) {
            throw new CSVAnnotationDeserializationError();
        }
    }

    public applyLabels(imageData: ImageData[], csvRows: CSVRow[]): ImportResult {
        const cleanImageData: ImageData[] = imageData.map((item: ImageData) => ImageDataUtil.cleanAnnotations(item));
        
        // Get existing labels
        const existingLabelNames: LabelName[] = LabelsSelector.getLabelNames();
        let labelNames: LabelName[] = [...existingLabelNames];
        
        // Create a map to store behavior name to label ID mapping
        const behaviorLabelMap: Map<string, string> = new Map();
        
        // Collect all unique behaviors from CSV
        const uniqueBehaviors = [...new Set(csvRows.map(row => row.behavior).filter(behavior => behavior))];
        
        // Create or find labels for each unique behavior
        for (const behaviorName of uniqueBehaviors) {
            const existingLabel = existingLabelNames.find(label => label.name === behaviorName);
            if (existingLabel) {
                behaviorLabelMap.set(behaviorName, existingLabel.id);
            } else {
                const newLabel = LabelUtil.createLabelName(behaviorName);
                labelNames.push(newLabel);
                behaviorLabelMap.set(behaviorName, newLabel.id);
            }
        }

        // Apply labels to each image based on CSV timestamp data
        for (const csvRow of csvRows) {
            if (!csvRow.behavior || csvRow.timestamp == null || csvRow.endTimestamp == null) continue;
            const labelId = behaviorLabelMap.get(csvRow.behavior);
            if (!labelId) continue;
            for (const targetImage of cleanImageData) {
                // compute timestamp strings and frames
                const fr = targetImage.frameRate ?? 30;
                const tsSec = csvRow.timestamp!;
                const endSec = csvRow.endTimestamp!;
                const timestampStr = tsSec.toFixed(3) + 's';
                const endTimestampStr = endSec.toFixed(3) + 's';
                const labelRect: LabelRect = {
                    id: uuidv4(),
                    labelId,
                    isVisible: true,
                    isCreatedByAI: false,
                    status: LabelStatus.ACCEPTED,
                    suggestedLabel: null,
                    timestamp: timestampStr,
                    endTimestamp: endTimestampStr,
                    behavior: csvRow.behavior
                };
                targetImage.labelRects.push(labelRect);
            }
        }

        return {
            imagesData: ImageDataUtil.arrange(cleanImageData, imageData.map((item: ImageData) => item.id)),
            labelNames: labelNames
        };
    }

    public static validateCSVFormat(header: string[]): void {
        const missingColumns = CSVImporter.requiredColumns.filter((col: string) => !header.includes(col));
        if (missingColumns.length !== 0) {
            throw new CSVFormatValidationError(`CSV file does not contain all required columns: ${missingColumns.join(', ')}`);
        }
    }
}
