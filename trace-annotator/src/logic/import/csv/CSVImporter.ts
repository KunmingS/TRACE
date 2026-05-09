import {ImageData, LabelName, LabelRect, Subject} from '../../../store/labels/types';
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
import {AnnotationImporter, ImportResult, ImportOnSuccess} from '../AnnotationImporter';
import {Settings} from "../../../settings/Settings";
import {DEFAULT_SUBJECT_ID} from '../../../store/labels/reducer';

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
    // Optional subject/animal name. Absent in legacy (pre-multi-animal) CSVs.
    animal?: string;
}

// Metadata extracted from the optional `# trace-meta:` comment line that
// precedes the column header. Currently only carries shortcut bindings;
// the format is intentionally extensible (space-separated key=value pairs).
export type CSVMeta = {
    behaviorShortcuts: Map<string, string>;  // name -> shortcut
}

// Parses a single `# trace-meta: key=value key=value` line into a CSVMeta.
// Today we only understand `behaviors=name1:s1,name2:s2`. Unknown keys are
// ignored so older parsers tolerate future additions.
function parseTraceMeta(line: string): CSVMeta {
    const meta: CSVMeta = { behaviorShortcuts: new Map() };
    const body = line.replace(/^#\s*trace-meta:\s*/i, '');
    // Tokenise on whitespace; values can contain commas/colons but no spaces.
    for (const token of body.split(/\s+/)) {
        if (!token) continue;
        const eq = token.indexOf('=');
        if (eq < 0) continue;
        const key = token.slice(0, eq).toLowerCase();
        const value = token.slice(eq + 1);
        if (key === 'behaviors' && value) {
            for (const pair of value.split(',')) {
                const colon = pair.indexOf(':');
                if (colon < 0) continue;
                const name = decodeURIComponent(pair.slice(0, colon));
                const shortcut = decodeURIComponent(pair.slice(colon + 1));
                if (name && shortcut) meta.behaviorShortcuts.set(name, shortcut);
            }
        }
    }
    return meta;
}

export class CSVImporter extends AnnotationImporter {
    public static requiredColumns = ['start_frame', 'end_frame', 'behavior']

    public import(
        filesData: File[],
        onSuccess: ImportOnSuccess,
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
                const {rows: csvRows, meta} = CSVImporter.parseCSVWithMeta(evt.target.result);
                const {imagesData, labelNames, subjects} = this.applyLabels(inputImagesData, csvRows, meta);
                onSuccess(imagesData, labelNames, subjects);
            } catch (error) {
                onFailure(error as Error);
            }
        };
        reader.onerror = () => onFailure(new CSVAnnotationReadingError());
    }

    // Back-compat shim — keeps old callers (e.g., tests) working.
    public static parseCSV(text: string): CSVRow[] {
        return CSVImporter.parseCSVWithMeta(text).rows;
    }

    public static parseCSVWithMeta(text: string): { rows: CSVRow[]; meta: CSVMeta } {
        try {
            const allLines = text.trim().split('\n');
            // Walk past leading comment lines (`# ...`). The first one matching
            // `# trace-meta:` becomes our metadata source; other comments are
            // tolerated and skipped so users can keep notes at the top of the file.
            let meta: CSVMeta = { behaviorShortcuts: new Map() };
            let headerIdx = 0;
            for (; headerIdx < allLines.length; headerIdx++) {
                const trimmed = allLines[headerIdx].trim();
                if (!trimmed.startsWith('#')) break;
                if (/^#\s*trace-meta:/i.test(trimmed)) {
                    meta = parseTraceMeta(trimmed);
                }
            }
            const lines = allLines.slice(headerIdx);
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
            // Optional `animal` column — present in CSVs from the multi-animal
            // version, absent in legacy CSVs.
            const idxAnimal = lower.indexOf('animal');

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
                if (idxAnimal >= 0 && values[idxAnimal]) {
                    row.animal = values[idxAnimal];
                }
                rows.push(row);
            }

            return { rows, meta };
        } catch (error) {
            throw new CSVAnnotationDeserializationError();
        }
    }

    public applyLabels(imageData: ImageData[], csvRows: CSVRow[], meta?: CSVMeta): ImportResult {
        const cleanImageData: ImageData[] = imageData.map((item: ImageData) => ImageDataUtil.cleanAnnotations(item));

        // Get existing labels
        const existingLabelNames: LabelName[] = LabelsSelector.getLabelNames();
        let labelNames: LabelName[] = [...existingLabelNames];

        // Create a map to store behavior name to label ID mapping
        const behaviorLabelMap: Map<string, string> = new Map();

        // Collect all unique behaviors from CSV
        const uniqueBehaviors = [...new Set(csvRows.map(row => row.behavior).filter(behavior => behavior))];

        // Create or find labels for each unique behavior. When the CSV
        // carries a `# trace-meta:` line, we honour the persisted shortcut for
        // each behavior — both for newly-materialised labels and for existing
        // ones that don't yet have a shortcut bound (e.g., re-importing a CSV
        // into a session where the shortcut was cleared).
        const metaShortcuts = meta?.behaviorShortcuts ?? new Map<string, string>();
        for (const behaviorName of uniqueBehaviors) {
            const metaShortcut = metaShortcuts.get(behaviorName);
            const existingLabel = existingLabelNames.find(label => label.name === behaviorName);
            if (existingLabel) {
                if (metaShortcut && !existingLabel.shortcut) {
                    // Mutate the in-memory copy we'll persist back via the
                    // returned `labelNames` array.
                    const idx = labelNames.findIndex(l => l.id === existingLabel.id);
                    if (idx >= 0) labelNames[idx] = { ...labelNames[idx], shortcut: metaShortcut };
                }
                behaviorLabelMap.set(behaviorName, existingLabel.id);
            } else {
                const newLabel = LabelUtil.createLabelName(behaviorName, metaShortcut);
                labelNames.push(newLabel);
                behaviorLabelMap.set(behaviorName, newLabel.id);
            }
        }

        // ── Subjects: materialize from the optional `animal` column ────
        const existingSubjects: Subject[] = LabelsSelector.getSubjects();
        let subjects: Subject[] = [...existingSubjects];
        const animalNameToId: Map<string, string> = new Map();
        for (const s of existingSubjects) animalNameToId.set(s.name, s.id);

        const uniqueAnimals = [...new Set(
            csvRows.map(row => row.animal).filter((a): a is string => !!a && a.trim() !== '')
        )];
        for (const animalName of uniqueAnimals) {
            if (animalNameToId.has(animalName)) continue;
            const subj = LabelUtil.createSubject(animalName);
            subjects.push(subj);
            animalNameToId.set(animalName, subj.id);
        }
        const fallbackSubjectId = subjects[0]?.id ?? DEFAULT_SUBJECT_ID;

        // Apply labels to each image based on CSV timestamp data
        for (const csvRow of csvRows) {
            if (!csvRow.behavior || csvRow.timestamp == null || csvRow.endTimestamp == null) continue;
            const labelId = behaviorLabelMap.get(csvRow.behavior);
            if (!labelId) continue;
            const animalId = csvRow.animal && animalNameToId.has(csvRow.animal)
                ? animalNameToId.get(csvRow.animal)!
                : fallbackSubjectId;
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
                    behavior: csvRow.behavior,
                    animalId
                };
                targetImage.labelRects.push(labelRect);
            }
        }

        return {
            imagesData: ImageDataUtil.arrange(cleanImageData, imageData.map((item: ImageData) => item.id)),
            labelNames: labelNames,
            subjects: subjects
        };
    }

    public static validateCSVFormat(header: string[]): void {
        const missingColumns = CSVImporter.requiredColumns.filter((col: string) => !header.includes(col));
        if (missingColumns.length !== 0) {
            throw new CSVFormatValidationError(`CSV file does not contain all required columns: ${missingColumns.join(', ')}`);
        }
    }
}
