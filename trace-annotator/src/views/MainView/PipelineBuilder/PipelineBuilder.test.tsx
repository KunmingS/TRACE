import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import PipelineBuilder from './PipelineBuilder';
import type { PipelineState } from './usePipelineState';

jest.mock('../../Common/PathPicker/PathPicker', () => ({
    __esModule: true,
    default: () => <div>Path Picker</div>,
}));

jest.mock('../../Common/LogViewer/LogViewer', () => ({
    __esModule: true,
    default: () => <div>Log Viewer</div>,
}));

jest.mock('./PipelineResults', () => ({
    __esModule: true,
    default: () => <div>Pipeline Results</div>,
}));

jest.mock('./JobDashboard', () => ({
    __esModule: true,
    default: () => <div>Job Dashboard</div>,
}));

function makePipeline(overrides: Partial<PipelineState> = {}): PipelineState {
    return {
        config: {
            steps: { train: false, test: false, infer: true },
            datasetSelection: { folder: '', stems: [], pairs: [], csvByStem: {} },
            testDatasetSelection: { folder: '', stems: [], pairs: [], csvByStem: {} },
            inputSelection: { folder: '/videos', stems: ['clip01'], pairs: [], csvByStem: {} },
            modelSize: 'small',
            modelLoadPath: '/models/model_20260507_120000',
            trainCacheMode: 'cached_video',
            trainCacheResolution: 144,
            resources: {
                train: { profile: 'balanced', batchSize: 4, numWorkers: 4, decodeThreads: 2, prefetchFactor: 2, advancedOpen: false },
                test: { profile: 'balanced', batchSize: 4, numWorkers: 4, decodeThreads: 2, prefetchFactor: 2, advancedOpen: false },
                infer: { profile: 'balanced', batchSize: 4, numWorkers: 4, decodeThreads: 2, prefetchFactor: 2, advancedOpen: false },
            },
            inferAnnotatedVideo: false,
            inferThreshold: 0.3,
            totalEpochs: 100,
            valStartEpoch: 50,
            valInterval: 10,
            trainValRatio: 0.8,
        },
        setConfig: jest.fn(),
        toggleStep: jest.fn(),
        pipelineStatus: 'idle',
        stepStates: {
            prep: { status: 'pending', jobId: null, error: null },
            train: { status: 'pending', jobId: null, error: null },
            test: { status: 'pending', jobId: null, error: null },
            infer: { status: 'pending', jobId: null, error: null },
        },
        activeJobId: null,
        currentStep: null,
        run: jest.fn(),
        cancel: jest.fn(),
        reset: jest.fn(),
        metrics: null,
        predictions: null,
        trainingEstimate: null,
        trainingEstimateStatus: 'idle',
        trainingEstimateError: null,
        canRun: true,
        validationError: null,
        cliCommand: null,
        cliCommandStatus: 'idle',
        cliCommandError: null,
        generateCliCommand: jest.fn(),
        ...overrides,
    };
}

describe('PipelineBuilder CLI command controls', () => {
    test('renders CLI Command next to Run Model and triggers generation', () => {
        const pipeline = makePipeline();
        render(<PipelineBuilder pipeline={pipeline} />);

        fireEvent.click(screen.getByRole('button', { name: /generate cli command/i }));

        expect(pipeline.generateCliCommand).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('button', { name: /run model/i })).toBeInTheDocument();
    });

    test('disables CLI Command when validation fails', () => {
        render(<PipelineBuilder pipeline={makePipeline({
            canRun: false,
            validationError: 'Pick at least one inference video.',
        })} />);

        expect(screen.getByRole('button', { name: /generate cli command/i })).toBeDisabled();
        expect(screen.getByText('Pick at least one inference video.')).toBeInTheDocument();
    });

    test('renders generated command panel', () => {
        render(<PipelineBuilder pipeline={makePipeline({
            cliCommandStatus: 'ready',
            cliCommand: {
                argv: ['trace', 'pipeline', '--infer'],
                command: 'trace pipeline --infer',
                warnings: [],
            },
        })} />);

        expect(screen.getByText('trace pipeline --infer')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument();
    });

    test('separates train, test, and predict setup into tabs', () => {
        render(<PipelineBuilder pipeline={makePipeline()} />);

        expect(screen.getAllByRole('tab')).toHaveLength(3);
        expect(screen.getByRole('tab', { name: /train model/i })).toHaveAttribute('aria-selected', 'true');

        fireEvent.click(screen.getByRole('tab', { name: /run predictions/i }));

        expect(screen.getByRole('tab', { name: /run predictions/i })).toHaveAttribute('aria-selected', 'true');
        expect(screen.getByText('Input Videos')).toBeInTheDocument();
    });

    test('shows a stage enable switch on disabled stage tabs', () => {
        const pipeline = makePipeline();
        render(<PipelineBuilder pipeline={pipeline} />);

        fireEvent.click(screen.getByRole('checkbox', { name: /enable train model/i }));

        expect(pipeline.toggleStep).toHaveBeenCalledWith('train');
    });

    test('keeps advanced prediction resources collapsed until opened', () => {
        const pipeline = makePipeline();
        const { rerender } = render(<PipelineBuilder pipeline={pipeline} />);

        fireEvent.click(screen.getByRole('tab', { name: /run predictions/i }));

        expect(screen.queryByLabelText(/prediction batch size/i)).not.toBeInTheDocument();
        rerender(<PipelineBuilder pipeline={{
            ...pipeline,
            config: {
                ...pipeline.config,
                resources: {
                    ...pipeline.config.resources,
                    infer: { ...pipeline.config.resources.infer, advancedOpen: true },
                },
            },
        }} />);

        expect(screen.getByLabelText(/prediction batch size/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/prediction workers/i)).toBeInTheDocument();
    });
});
