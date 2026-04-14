export interface IEditorFeature {
    displayText: string;
    imageSrc: string;
    imageAlt: string;
}

export const EditorFeatureData: IEditorFeature[] = [
    {
        displayText: 'Open source and free to use',
        imageSrc: 'ico/open-source.png',
        imageAlt: 'open-source'
    },
    {
        displayText: 'No need to send your data anywhere',
        imageSrc: 'ico/private.png',
        imageAlt: 'private'
    },
    {
        displayText: 'Support for multiple annotation types',
        imageSrc: 'ico/labels.png',
        imageAlt: 'labels'
    },
    {
        displayText: 'AI-powered suggestions',
        imageSrc: 'ico/ai.png',
        imageAlt: 'ai'
    },
    {
        displayText: 'Export in multiple formats',
        imageSrc: 'ico/export.png',
        imageAlt: 'export'
    },
    {
        displayText: 'Works on any modern browser',
        imageSrc: 'ico/browser.png',
        imageAlt: 'browser'
    }
];
