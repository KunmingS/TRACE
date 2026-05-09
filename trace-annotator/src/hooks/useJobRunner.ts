import { useCallback, useEffect, useRef, useState } from 'react';
import { API_URL } from '../config';

export type JobStatus = 'idle' | 'submitting' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface UseJobRunnerReturn {
    jobId: string | null;
    status: JobStatus;
    error: string | null;
    submit: (jobType: 'prep' | 'train' | 'test' | 'infer', body: Record<string, any>) => Promise<string>;
    cancel: () => Promise<void>;
    fetchArtifact: <T>(filename: string) => Promise<T | null>;
    reset: () => void;
}

const POLL_INTERVAL_MS = 2500;

export function useJobRunner(): UseJobRunnerReturn {
    const [jobId, setJobId] = useState<string | null>(null);
    const [status, setStatus] = useState<JobStatus>('idle');
    const [error, setError] = useState<string | null>(null);

    const pollRef = useRef<number | null>(null);
    const jobIdRef = useRef<string | null>(null);

    // Keep ref in sync for use in callbacks
    useEffect(() => {
        jobIdRef.current = jobId;
    }, [jobId]);

    // Cleanup polling on unmount
    useEffect(() => {
        return () => {
            if (pollRef.current) {
                clearInterval(pollRef.current);
                pollRef.current = null;
            }
        };
    }, []);

    const stopPolling = useCallback(() => {
        if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
        }
    }, []);

    const startPolling = useCallback((id: string) => {
        stopPolling();
        pollRef.current = window.setInterval(async () => {
            try {
                const res = await fetch(`${API_URL}/api/jobs/${id}`);
                if (!res.ok) return;
                const job = await res.json();
                if (job.status === 'completed') {
                    stopPolling();
                    setStatus('completed');
                } else if (job.status === 'failed') {
                    stopPolling();
                    setStatus('failed');
                    setError(job.error_message || 'Job failed');
                } else if (job.status === 'cancelled') {
                    stopPolling();
                    setStatus('cancelled');
                }
            } catch {
                // Network error — keep polling
            }
        }, POLL_INTERVAL_MS);
    }, [stopPolling]);

    const submit = useCallback(async (
        jobType: 'prep' | 'train' | 'test' | 'infer',
        body: Record<string, any>,
    ): Promise<string> => {
        setError(null);
        setStatus('submitting');

        const res = await fetch(`${API_URL}/api/jobs/${jobType}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            const body = await res.json().catch(() => ({}));
            const raw = body.detail;
            const msg = typeof raw === 'string'
                ? raw
                : (raw?.message || `Job submission failed (${res.status})`);
            setStatus('failed');
            setError(msg);
            throw new Error(msg);
        }

        const job = await res.json();
        const id = job.job_id;
        setJobId(id);
        setStatus('running');
        startPolling(id);
        return id;
    }, [startPolling]);

    const cancel = useCallback(async () => {
        const id = jobIdRef.current;
        if (!id) return;
        stopPolling();
        try {
            await fetch(`${API_URL}/api/jobs/${id}/cancel`, { method: 'POST' });
        } catch {
            // Best-effort
        }
        setStatus('cancelled');
    }, [stopPolling]);

    const fetchArtifact = useCallback(async <T,>(filename: string): Promise<T | null> => {
        const id = jobIdRef.current;
        if (!id) return null;
        try {
            const res = await fetch(`${API_URL}/api/jobs/${id}/artifacts/${filename}`);
            if (!res.ok) return null;
            return await res.json() as T;
        } catch {
            return null;
        }
    }, []);

    const reset = useCallback(() => {
        stopPolling();
        setJobId(null);
        setStatus('idle');
        setError(null);
    }, [stopPolling]);

    return { jobId, status, error, submit, cancel, fetchArtifact, reset };
}

/**
 * Wait for a job runner to reach a terminal state.
 * Returns the final status. Useful for pipeline sequencing.
 */
export function waitForTerminal(runner: UseJobRunnerReturn): Promise<JobStatus> {
    return new Promise((resolve) => {
        const check = () => {
            const s = runner.status;
            if (s === 'completed' || s === 'failed' || s === 'cancelled') {
                resolve(s);
            } else {
                setTimeout(check, 500);
            }
        };
        check();
    });
}
