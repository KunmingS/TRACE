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
    default: () => <div>Run Model Panel</div>,
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

// Workspace tab buttons are rendered with `role="tab"` (inside a `role="tablist"`)
// for accessibility, so we look them up by `aria-label` rather than the
// implicit `button` role — the explicit role attribute wins for a11y queries.
const tabByName = (name: 'Annotate' | 'Run Model' | 'Tutorial') =>
    screen.getByRole('tab', { name: `${name} tab` });

describe('MainView', () => {
    test('shows Annotate panel by default', () => {
        render(<MainView />);
        expect(screen.getByText('Annotate Panel')).toBeInTheDocument();
    });

    test('switches to Run Model panel when Run Model mode is clicked', () => {
        render(<MainView />);
        fireEvent.click(tabByName('Run Model'));
        expect(screen.getByText('Run Model Panel')).toBeInTheDocument();
    });

    test('switches back to Annotate when Annotate mode is clicked', () => {
        render(<MainView />);

        fireEvent.click(tabByName('Run Model'));
        expect(screen.getByText('Run Model Panel')).toBeInTheDocument();

        fireEvent.click(tabByName('Annotate'));
        expect(screen.getByText('Annotate Panel')).toBeInTheDocument();
    });

    test('keeps run step controls inside the Run Model panel', () => {
        render(<MainView />);
        fireEvent.click(tabByName('Run Model'));

        expect(screen.queryByLabelText('Enable training step')).not.toBeInTheDocument();
        expect(screen.queryByLabelText('Enable test step')).not.toBeInTheDocument();
        expect(screen.queryByLabelText('Enable prediction step')).not.toBeInTheDocument();
    });

    test('switches to Tutorial panel when Tutorial tab is clicked', () => {
        render(<MainView />);
        fireEvent.click(tabByName('Tutorial'));
        expect(screen.getByText('Tutorial Panel')).toBeInTheDocument();
    });

    test('does not show step toggles in tutorial mode', () => {
        render(<MainView />);
        fireEvent.click(tabByName('Tutorial'));

        expect(screen.queryByText('Train')).not.toBeInTheDocument();
        expect(screen.queryByText('Test')).not.toBeInTheDocument();
        expect(screen.queryByText('Predict')).not.toBeInTheDocument();
    });

    test('renders three workspace tabs', () => {
        render(<MainView />);
        expect(screen.getAllByRole('tab')).toHaveLength(3);
    });
});
