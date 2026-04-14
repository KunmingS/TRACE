export class CSVAnnotationFileCountError extends Error {
    constructor() {
        super('CSV annotation import accepts only one file.');
        Object.setPrototypeOf(this, CSVAnnotationFileCountError.prototype);
    }
}

export class CSVAnnotationReadingError extends Error {
    constructor() {
        super('Problem with reading CSV annotation file.');
        Object.setPrototypeOf(this, CSVAnnotationReadingError.prototype);
    }
}

export class CSVAnnotationDeserializationError extends Error {
    constructor() {
        super('Problem with CSV annotation file deserialization.');
        Object.setPrototypeOf(this, CSVAnnotationDeserializationError.prototype);
    }
}

export class CSVFormatValidationError extends Error {
    constructor(message?: string) {
        super(message || 'CSV format validation failed.');
        Object.setPrototypeOf(this, CSVFormatValidationError.prototype);
    }
}
