import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import TestPanel from './TestPanel';

jest.mock('../../../config', () => ({ API_URL: '' }));

jest.mock('../../Common/PathPicker/PathPicker', () => ({
    __esModule: true,
    default: ({ value, onChange, placeholder, disabled }: any) => (
        <input
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder={placeholder}
            disabled={disabled}
        />
    ),
}));

jest.mock('../../Common/LogViewer/LogViewer', () => ({
    __esModule: true,
    default: ({ jobId }: any) => <div data-testid='log-viewer'>{jobId}</div>,
}));

const jsonResponse = (body: any) => Promise.resolve({
    ok: true,
    status: 200,
    json: async () => body,
});

describe('TestPanel', () => {
    beforeEach(() => {
        jest.useFakeTimers();
        global.fetch = jest.fn() as any;
    });

    afterEach(() => {
        jest.runOnlyPendingTimers();
        jest.useRealTimers();
        jest.resetAllMocks();
    });

    test('renders model list and evaluation form', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock.mockImplementationOnce(() =>
            jsonResponse([
                {
                    path: '/models/dev-model',
                    label: 'dev-model',
                    config_path: 'configs/large.py',
                    classes: ['drink', 'eat'],
                    num_classes: 2,
                    size_mb: 48.2,
                },
            ])
        );

        render(<TestPanel />);

        // Wait for models to load
        await waitFor(() => {
            expect(screen.getByRole('button', { name: /dev-model/i })).toBeInTheDocument();
        });

        // Enter dataset path
        fireEvent.change(screen.getByPlaceholderText('/path/to/dataset'), {
            target: { value: '/datasets/raw' },
        });

        // Select model
        fireEvent.click(screen.getByRole('button', { name: /dev-model/i }));

        // Run Evaluation button should be enabled
        expect(screen.getByRole('button', { name: 'Run Evaluation' })).not.toBeDisabled();
    });

    test('submits evaluation job with correct parameters', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock
            // Initial models load
            .mockImplementationOnce(() =>
                jsonResponse([
                    {
                        path: '/models/dev-model',
                        label: 'dev-model',
                        config_path: 'configs/large.py',
                        classes: ['drink', 'eat'],
                        num_classes: 2,
                        size_mb: 48.2,
                    },
                ])
            )
            // Prep job submission
            .mockImplementationOnce(() => jsonResponse({ job_id: 'prep-1' }))
            // Prep status poll - completed
            .mockImplementationOnce(() => jsonResponse({ status: 'completed' }))
            // Prep artifact
            .mockImplementationOnce(() =>
                jsonResponse({
                    model_dir: '/datasets/raw/model_20260507_120000',
                    dataset_json: '/datasets/raw/model_20260507_120000/dataset.json',
                    classmap_path: '/datasets/raw/model_20260507_120000/classmap.txt',
                })
            )
            // Test job submission
            .mockImplementationOnce(() => jsonResponse({ job_id: 'test-1' }));

        render(<TestPanel />);

        fireEvent.change(await screen.findByPlaceholderText('/path/to/dataset'), {
            target: { value: '/datasets/raw' },
        });
        fireEvent.click(await screen.findByRole('button', { name: /dev-model/i }));

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'Run Evaluation' }));
        });

        // Advance past prep polling
        await act(async () => {
            jest.advanceTimersByTime(2000);
            await Promise.resolve();
        });

        // Verify test job was submitted
        await waitFor(() => {
            const testCall = fetchMock.mock.calls.find(
                (c: any) => typeof c[0] === 'string' && c[0].includes('/api/jobs/test')
            );
            expect(testCall).toBeTruthy();
            const testRequest = JSON.parse(testCall[1].body);
            expect(testRequest).toMatchObject({
                model_dir: '/models/dev-model',
                dataset_dir: '/datasets/raw/model_20260507_120000',
                annotation_path: '/datasets/raw/model_20260507_120000/dataset.json',
            });
        });
    });

    test('shows CUDA-unavailable error when evaluation submission rejects', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock
            .mockImplementationOnce(() =>
                jsonResponse([
                    {
                        path: '/models/dev-model',
                        label: 'dev-model',
                        config_path: 'configs/large.py',
                        classes: ['drink', 'eat'],
                        num_classes: 2,
                        size_mb: 48.2,
                    },
                ])
            )
            .mockImplementationOnce(() => jsonResponse({ job_id: 'prep-1' }))
            .mockImplementationOnce(() => jsonResponse({ status: 'completed' }))
            .mockImplementationOnce(() =>
                jsonResponse({
                    model_dir: '/datasets/raw/model_20260507_120000',
                    dataset_json: '/datasets/raw/model_20260507_120000/dataset.json',
                    classmap_path: '/datasets/raw/model_20260507_120000/classmap.txt',
                })
            )
            .mockImplementationOnce(() => Promise.resolve({
                ok: false,
                status: 400,
                json: async () => ({
                    detail: { code: 'CUDA_UNAVAILABLE', message: 'No GPU on server.' },
                }),
            }));

        render(<TestPanel />);

        fireEvent.change(await screen.findByPlaceholderText('/path/to/dataset'), {
            target: { value: '/datasets/raw' },
        });
        fireEvent.click(await screen.findByRole('button', { name: /dev-model/i }));

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'Run Evaluation' }));
        });

        await act(async () => {
            jest.advanceTimersByTime(2000);
            await Promise.resolve();
        });

        expect(await screen.findByText('No GPU on server.')).toBeInTheDocument();
    });
});
