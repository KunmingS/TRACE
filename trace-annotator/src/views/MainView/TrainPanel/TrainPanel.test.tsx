import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import TrainPanel from './TrainPanel';

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

describe('TrainPanel', () => {
    beforeEach(() => {
        jest.useFakeTimers();
        global.fetch = jest.fn() as any;
    });

    afterEach(() => {
        jest.runOnlyPendingTimers();
        jest.useRealTimers();
        jest.resetAllMocks();
    });

    test('prepares a dataset and submits training with prepared artifact paths', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock
            .mockImplementationOnce(() =>
                jsonResponse([{ path: 'configs/tridet/tridet_small.py', name: 'tridet_small' }])
            )
            .mockImplementationOnce(() => jsonResponse({ job_id: 'prep-1' }))
            .mockImplementationOnce(() => jsonResponse({ status: 'completed' }))
            .mockImplementationOnce(() =>
                jsonResponse({
                    clips_dir: '/prepared/clips',
                    json_path: '/prepared/dataset.json',
                    classmap_path: '/prepared/classmap.txt',
                })
            )
            .mockImplementationOnce(() => jsonResponse({ job_id: 'train-1' }));

        render(<TrainPanel />);

        fireEvent.change(
            await screen.findByPlaceholderText('/path/to/dataset (videos + CSVs)'),
            { target: { value: '/datasets/raw' } }
        );
        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'Prepare' }));
        });

        await waitFor(() => {
            expect(fetchMock).toHaveBeenCalledWith(
                '/api/jobs/prep',
                expect.objectContaining({
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                })
            );
        });

        await act(async () => {
            jest.advanceTimersByTime(2000);
            await Promise.resolve();
        });

        await screen.findByText('Dataset ready');

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /Start Training/i }));
        });

        await waitFor(() => {
            expect(fetchMock).toHaveBeenCalledTimes(5);
        });

        const trainRequest = JSON.parse((fetchMock.mock.calls[4][1] as any).body);
        expect(trainRequest).toMatchObject({
            config_path: 'configs/tridet/tridet_small.py',
            dataset_dir: '/prepared/clips',
            annotation_path: '/prepared/dataset.json',
            class_map: '/prepared/classmap.txt',
        });
        expect(typeof trainRequest.exp_id).toBe('number');
    });
});
