import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import LogViewer from './LogViewer';

jest.mock('../../../config', () => ({ API_URL: '' }));

const jsonResponse = (body: any) => Promise.resolve({
    ok: true,
    json: async () => body,
});

class MockEventSource {
    static instances: MockEventSource[] = [];

    public onmessage: ((event: MessageEvent) => void) | null = null;
    public onerror: ((event: Event) => void) | null = null;
    public close = jest.fn();
    public url: string;

    constructor(url: string) {
        this.url = url;
        MockEventSource.instances.push(this);
    }
}

describe('LogViewer', () => {
    beforeEach(() => {
        MockEventSource.instances = [];
        global.fetch = jest.fn() as any;
        (global as any).EventSource = MockEventSource;
    });

    afterEach(() => {
        jest.resetAllMocks();
    });

    test('streams log lines, strips ANSI codes, and fetches final status after stream closes', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock.mockImplementationOnce(() => jsonResponse({ status: 'completed' }));

        render(<LogViewer jobId='job-1' />);

        expect(MockEventSource.instances[0].url).toBe('/api/jobs/job-1/logs/stream');

        await act(async () => {
            MockEventSource.instances[0].onmessage?.({ data: '\u001b[31mtraining...\u001b[0m' } as MessageEvent);
            await Promise.resolve();
        });
        expect(await screen.findByText('training...')).toBeInTheDocument();

        await act(async () => {
            MockEventSource.instances[0].onerror?.(new Event('error'));
            await Promise.resolve();
        });

        await screen.findByText('completed');
        expect(fetchMock).toHaveBeenCalledWith('/api/jobs/job-1');
    });

    test('posts cancel requests while the log is live', async () => {
        const fetchMock = global.fetch as jest.Mock;
        const onCancel = jest.fn();

        render(<LogViewer jobId='job-2' onCancel={onCancel} />);

        fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));

        await waitFor(() => {
            expect(fetchMock).toHaveBeenCalledWith('/api/jobs/job-2/cancel', { method: 'POST' });
        });
        expect(onCancel).toHaveBeenCalled();
    });
});
