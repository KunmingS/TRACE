import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import TrainPanel from './TrainPanel';

jest.mock('../../../config', () => ({ API_URL: '' }));

jest.mock('../../Common/PathPicker/PathPicker', () => ({
    __esModule: true,
    default: ({ value, onChange, onSelectedStemsChange, onSelectedPairsChange, placeholder, disabled }: any) => (
        <input
            value={value}
            onChange={(event) => {
                onChange(event.target.value);
                onSelectedStemsChange?.(['session01']);
                onSelectedPairsChange?.(['session01.mp4=session01.csv']);
            }}
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

    test('prepares a dataset and submits training with model artifact paths', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock
            .mockImplementationOnce(() =>
                jsonResponse([{ path: 'configs/small.py', name: 'small' }])
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
        const prepRequest = JSON.parse((fetchMock.mock.calls[1][1] as any).body);
        expect(prepRequest).toMatchObject({
            work_dir: '/datasets/raw',
            explicit_pairs: ['session01.mp4=session01.csv'],
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
            config_path: 'configs/small.py',
            model_dir: '/datasets/raw/model_20260507_120000',
            dataset_dir: '/datasets/raw/model_20260507_120000',
            annotation_path: '/datasets/raw/model_20260507_120000/dataset.json',
            class_map: '/datasets/raw/model_20260507_120000/classmap.txt',
        });
    });

    test('shows CUDA-unavailable error when training submission rejects', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock
            .mockImplementationOnce(() =>
                jsonResponse([{ path: 'configs/small.py', name: 'small' }])
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

        render(<TrainPanel />);

        fireEvent.change(
            await screen.findByPlaceholderText('/path/to/dataset (videos + CSVs)'),
            { target: { value: '/datasets/raw' } }
        );
        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'Prepare' }));
        });

        await act(async () => {
            jest.advanceTimersByTime(2000);
            await Promise.resolve();
        });

        await screen.findByText('Dataset ready');

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /Start Training/i }));
        });

        expect(await screen.findByText('No GPU on server.')).toBeInTheDocument();
    });
});
