import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import JobDashboard from './JobDashboard';

jest.mock('../../../config', () => ({ API_URL: '' }));

function jsonResponse(body: unknown) {
    return {
        ok: true,
        json: async () => body,
    };
}

class MockEventSource {
    static instances: MockEventSource[] = [];
    public onmessage: ((event: MessageEvent) => void) | null = null;
    public onerror: ((event: Event) => void) | null = null;
    public close = jest.fn();

    constructor(public url: string) {
        MockEventSource.instances.push(this);
    }
}

describe('JobDashboard backend run grouping', () => {
    beforeEach(() => {
        jest.useFakeTimers();
        MockEventSource.instances = [];
        global.fetch = jest.fn() as any;
        (global as any).EventSource = MockEventSource;
    });

    afterEach(() => {
        jest.useRealTimers();
        jest.resetAllMocks();
        MockEventSource.instances = [];
    });

    test('groups prep and train jobs by backend run_id and opens the train log', async () => {
        const prepJob = {
            job_id: '59d57a3ac4e0',
            job_type: 'prep',
            run_id: 'run-20260508-160136',
            stage: 'prep',
            run_steps: ['train'],
            status: 'completed',
            config_path: '/tank/rui/C0213159/',
            created_at: '2026-05-08T23:01:36.040033+00:00',
            started_at: '2026-05-08T23:01:36.042440+00:00',
            finished_at: '2026-05-08T23:02:55.178246+00:00',
            pid: 3444585,
            log_file: '/tank/rui/C0213159/model_20260508_160136/prep.log',
            work_dir: '/tank/rui/C0213159/model_20260508_160136',
            error_message: null,
            args: {
                model_dir: '/tank/rui/C0213159/model_20260508_160136',
            },
        };
        const trainJob = {
            job_id: 'df6453365513',
            job_type: 'train',
            run_id: 'run-20260508-160136',
            stage: 'train',
            run_steps: ['train'],
            status: 'completed',
            config_path: 'configs/small.py',
            created_at: '2026-05-08T23:02:57.973958+00:00',
            started_at: '2026-05-08T23:02:57.976310+00:00',
            finished_at: '2026-05-08T23:20:31.274761+00:00',
            pid: 3448319,
            log_file: '/tank/rui/C0213159/model_20260508_160136/job.log',
            work_dir: '/tank/rui/C0213159/model_20260508_160136',
            error_message: null,
            args: {
                model_dir: '/tank/rui/C0213159/model_20260508_160136',
            },
        };

        (global.fetch as jest.Mock).mockResolvedValue(jsonResponse([trainJob, prepJob]));

        const { unmount } = render(<JobDashboard />);

        await waitFor(() => expect(screen.getByText('Train')).toBeInTheDocument());
        expect(screen.getByText('2 tasks')).toBeInTheDocument();

        fireEvent.click(screen.getByText('Train'));

        expect(screen.getByText('59d57a3ac4')).toBeInTheDocument();
        await waitFor(() => expect(screen.getByText('df64533655')).toBeInTheDocument());

        fireEvent.click(screen.getByText('df64533655'));

        expect(screen.getByText('run-20260508-160136')).toBeInTheDocument();
        expect(screen.getAllByText('train').length).toBeGreaterThan(0);
        expect(screen.getByText('/tank/rui/C0213159/model_20260508_160136/job.log')).toBeInTheDocument();
        expect(MockEventSource.instances[0].url).toBe('/api/jobs/df6453365513/logs/stream');

        unmount();
    });

    test('shows annotated video render progress from streamed log lines', async () => {
        const inferJob = {
            job_id: 'bf95ae01bcaf',
            job_type: 'infer',
            run_id: 'bf95ae01bcaf',
            stage: 'infer',
            run_steps: ['infer'],
            status: 'running',
            config_path: 'configs/small.py',
            created_at: '2026-05-08T23:01:36.040033+00:00',
            started_at: '2026-05-08T23:01:36.042440+00:00',
            finished_at: null,
            pid: 3924406,
            log_file: '/tank/rui/predict/job.log',
            work_dir: '/tank/rui/predict',
            error_message: null,
            args: {
                annotated_video: true,
                auto_tune: false,
                included_stems: ['trial'],
                input: '/tank/rui/',
                output_dir: '/tank/rui/predict',
                profile: false,
                threshold: 0.3,
            },
        };

        (global.fetch as jest.Mock).mockResolvedValue(jsonResponse([inferJob]));

        render(<JobDashboard />);

        await waitFor(() => expect(screen.getByText('bf95ae01bc')).toBeInTheDocument());
        fireEvent.click(screen.getByText('bf95ae01bc'));

        expect(screen.getByText('Inference Input')).toBeInTheDocument();
        expect(screen.getByText('/tank/rui/')).toBeInTheDocument();
        expect(screen.getByText('Videos')).toBeInTheDocument();
        expect(screen.getAllByText('trial').length).toBeGreaterThan(0);
        expect(screen.getByText('Annotated Video')).toBeInTheDocument();
        expect(screen.getByText('/tank/rui/predict/trial_annotated.mp4')).toBeInTheDocument();
        expect(screen.queryByText('profile')).not.toBeInTheDocument();
        expect(screen.queryByText('auto_tune')).not.toBeInTheDocument();
        expect(screen.queryByText('annotated_video')).not.toBeInTheDocument();

        act(() => {
            MockEventSource.instances[0].onmessage?.({
                data: '2026-05-08 22:10:00 Infer INFO: Annotated render starts: trial (800 frames, 3840x1080)',
            } as MessageEvent);
            MockEventSource.instances[0].onmessage?.({
                data: '2026-05-08 22:10:05 Infer INFO: Annotated render progress: trial 300/800 frames (37.5%, 21.4 fps, eta 23s)',
            } as MessageEvent);
        });

        expect(screen.getByText('Rendering annotated video')).toBeInTheDocument();
        expect(screen.getByText('38%')).toBeInTheDocument();
        expect(screen.getAllByText('trial').length).toBeGreaterThan(1);
        expect(screen.getByText('300/800 frames, 21.4 fps, eta 23s')).toBeInTheDocument();
        expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '38');

        act(() => {
            MockEventSource.instances[0].onmessage?.({
                data: '2026-05-08 22:10:20 Infer INFO: Annotated render complete: trial (800/800 frames)',
            } as MessageEvent);
        });

        expect(screen.getByText('Annotated video ready')).toBeInTheDocument();
        expect(screen.getByText('100%')).toBeInTheDocument();
    });
});
