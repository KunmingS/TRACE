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
                    config_path: 'configs/tridet/tridet_large.py',
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
                        config_path: 'configs/tridet/tridet_large.py',
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
                    clips_dir: '/prepared/clips',
                    json_path: '/prepared/dataset.json',
                    classmap_path: '/prepared/classmap.txt',
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
                config_path: 'configs/tridet/tridet_large.py',
                checkpoint: '/models/dev-model/best.pth',
                class_map: '/models/dev-model/classmap.txt',
                dataset_dir: '/prepared/clips',
                annotation_path: '/prepared/dataset.json',
            });
        });
    });
});
