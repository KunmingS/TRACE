import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import MainView from './MainView';

jest.mock('react-redux', () => {
    const ReactLib = require('react');

    return {
        connect: () => (Component: React.ComponentType<any>) => {
            const Connected = (props: any) => {
                const [homeTab, setHomeTab] = ReactLib.useState('annotate');

                return <Component {...props} homeTab={homeTab} updateHomeTab={setHomeTab} />;
            };

            Connected.displayName = 'MockConnectedMainView';

            return Connected;
        },
    };
});

jest.mock('./ImagesDropZone/ImagesDropZone', () => ({
    __esModule: true,
    default: () => <div>Annotate Panel</div>,
}));

jest.mock('./PipelineBuilder/PipelineBuilder', () => ({
    __esModule: true,
    default: () => <div>Pipeline Panel</div>,
}));

jest.mock('./TutorialPanel/TutorialPanel', () => ({
    __esModule: true,
    default: () => <div>Tutorial Panel</div>,
}));

jest.mock('./PipelineBuilder/usePipelineState', () => ({
    usePipelineState: () => ({
        config: { steps: { train: true, test: true, infer: false }, datasetPath: '', modelSize: 'small', modelPath: '', inputPath: '' },
        setConfig: jest.fn(),
        toggleStep: jest.fn(),
        pipelineStatus: 'idle',
        stepStates: {},
        activeJobId: null,
        currentStep: null,
        run: jest.fn(),
        cancel: jest.fn(),
        reset: jest.fn(),
        metrics: null,
        predictions: null,
        canRun: false,
        validationError: null,
    }),
}));

describe('MainView', () => {
    test('shows Annotate panel by default', () => {
        render(<MainView />);
        expect(screen.getByText('Annotate Panel')).toBeInTheDocument();
    });

    test('switches to Pipeline panel when Pipeline mode is clicked', () => {
        render(<MainView />);
        const pipelineBtn = screen
            .getAllByRole('button')
            .find((btn) => btn.textContent?.includes('Pipeline')) as HTMLButtonElement;

        fireEvent.click(pipelineBtn);
        expect(screen.getByText('Pipeline Panel')).toBeInTheDocument();
    });

    test('switches back to Annotate when Annotate mode is clicked', () => {
        render(<MainView />);

        // Go to pipeline
        const pipelineBtn = screen
            .getAllByRole('button')
            .find((btn) => btn.textContent?.includes('Pipeline')) as HTMLButtonElement;
        fireEvent.click(pipelineBtn);
        expect(screen.getByText('Pipeline Panel')).toBeInTheDocument();

        // Go back to annotate
        const annotateBtn = screen
            .getAllByRole('button')
            .find((btn) => btn.textContent?.includes('Annotate')) as HTMLButtonElement;
        fireEvent.click(annotateBtn);
        expect(screen.getByText('Annotate Panel')).toBeInTheDocument();
    });

    test('shows step toggles when in pipeline mode', () => {
        render(<MainView />);
        const pipelineBtn = screen
            .getAllByRole('button')
            .find((btn) => btn.textContent?.includes('Pipeline')) as HTMLButtonElement;
        fireEvent.click(pipelineBtn);

        expect(screen.getByText('Train')).toBeInTheDocument();
        expect(screen.getByText('Test')).toBeInTheDocument();
        expect(screen.getByText('Inference')).toBeInTheDocument();
    });

    test('switches to Tutorial panel when Tutorial tab is clicked', () => {
        render(<MainView />);
        const tutorialBtn = screen
            .getAllByRole('button')
            .find((btn) => btn.textContent?.includes('Tutorial')) as HTMLButtonElement;
        fireEvent.click(tutorialBtn);
        expect(screen.getByText('Tutorial Panel')).toBeInTheDocument();
    });

    test('does not show step toggles in tutorial mode', () => {
        render(<MainView />);
        const tutorialBtn = screen
            .getAllByRole('button')
            .find((btn) => btn.textContent?.includes('Tutorial')) as HTMLButtonElement;
        fireEvent.click(tutorialBtn);

        expect(screen.queryByText('Train')).not.toBeInTheDocument();
        expect(screen.queryByText('Test')).not.toBeInTheDocument();
        expect(screen.queryByText('Inference')).not.toBeInTheDocument();
    });

    test('renders three workspace tabs', () => {
        render(<MainView />);
        const tabs = screen.getAllByRole('button').filter((btn) =>
            ['Annotate', 'Pipeline', 'Tutorial'].some((t) => btn.textContent?.includes(t))
        );
        expect(tabs).toHaveLength(3);
    });
});
