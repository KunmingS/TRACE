import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import InferPanel from './InferPanel';

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

describe('InferPanel', () => {
    beforeEach(() => {
        jest.useFakeTimers();
        global.fetch = jest.fn() as any;
    });

    afterEach(() => {
        jest.runOnlyPendingTimers();
        jest.useRealTimers();
        jest.resetAllMocks();
    });

    test('submits inference and renders returned detections', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock
            .mockImplementationOnce(() =>
                jsonResponse([
                    {
                        path: '/models/dev-model',
                        label: 'dev-model',
                        config_path: 'configs/tridet/tridet_small.py',
                        classes: ['drink', 'eat'],
                        num_classes: 2,
                        size_mb: 48.2,
                    },
                ])
            )
            .mockImplementationOnce(() => jsonResponse({ job_id: 'infer-1' }))
            .mockImplementationOnce(() => jsonResponse({ status: 'completed' }))
            .mockImplementationOnce(() =>
                jsonResponse({
                    predictions: {
                        clip_001: [
                            { segment: [0.5, 1.5], label: 'drink', score: 0.91 },
                            { segment: [2.0, 3.2], label: 'eat', score: 0.82 },
                        ],
                    },
                })
            );

        render(<InferPanel />);

        fireEvent.click(await screen.findByRole('button', { name: /dev-model/i }));
        fireEvent.change(screen.getByPlaceholderText('/path/to/video.mp4 or /path/to/folder'), {
            target: { value: '/videos/clip_001.mp4' },
        });
        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /Run Inference/i }));
        });

        await act(async () => {
            jest.advanceTimersByTime(3000);
            await Promise.resolve();
        });

        await waitFor(() => {
            expect(screen.getByText('Predictions')).toBeInTheDocument();
        });
        expect(screen.getByText('clip_001')).toBeInTheDocument();
        expect(screen.getByText('0.5s - 1.5s')).toBeInTheDocument();
        expect(screen.getByText('drink')).toBeInTheDocument();
        expect(screen.getByTestId('log-viewer')).toHaveTextContent('infer-1');

        const inferRequest = JSON.parse((fetchMock.mock.calls[1][1] as any).body);
        expect(inferRequest).toMatchObject({
            config_path: 'configs/tridet/tridet_small.py',
            checkpoint: '/models/dev-model/best.pth',
            class_map: '/models/dev-model/classmap.txt',
            input: '/videos/clip_001.mp4',
        });
        expect(typeof inferRequest.exp_id).toBe('number');
    });
});
