import React, { useCallback, useEffect, useRef, useState } from 'react';
import { API_URL } from '../../../config';

interface JobInfo {
    job_id: string;
    job_type: string;
    status: string;
    config_path: string | null;
    created_at: string;
    started_at: string | null;
    finished_at: string | null;
    pid: number | null;
    error_message: string | null;
    args?: Record<string, any>;
}

const POLL_INTERVAL = 4000;

const STATUS_ORDER: Record<string, number> = {
    running: 0,
    pending: 1,
    failed: 2,
    cancelled: 3,
    completed: 4,
};

function relativeTime(iso: string): string {
    const diff = Date.now() - new Date(iso).getTime();
    const sec = Math.floor(diff / 1000);
    if (sec < 60) return `${sec}s ago`;
    const min = Math.floor(sec / 60);
    if (min < 60) return `${min}m ago`;
    const hr = Math.floor(min / 60);
    if (hr < 24) return `${hr}h ago`;
    const d = Math.floor(hr / 24);
    return `${d}d ago`;
}

function formatTime(iso: string | null): string {
    if (!iso) return '-';
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
}

function duration(start: string | null, end: string | null): string {
    if (!start) return '-';
    const s = new Date(start).getTime();
    const e = end ? new Date(end).getTime() : Date.now();
    const sec = Math.floor((e - s) / 1000);
    if (sec < 60) return `${sec}s`;
    const min = Math.floor(sec / 60);
    const remSec = sec % 60;
    if (min < 60) return `${min}m ${remSec}s`;
    const hr = Math.floor(min / 60);
    return `${hr}h ${min % 60}m`;
}

/* ── Job Log Viewer (reads stored logs for past jobs) ── */
const JobLogPanel: React.FC<{ jobId: string }> = ({ jobId }) => {
    const [lines, setLines] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        setLines([]);
        setLoading(true);
        const collected: string[] = [];
        const es = new EventSource(`${API_URL}/api/jobs/${jobId}/logs/stream`);

        es.onmessage = (event) => {
            collected.push(event.data.replace(/\x1b\[[0-9;]*m/g, ''));
            // Batch updates for performance
            if (collected.length % 20 === 0 || collected.length < 5) {
                setLines([...collected]);
            }
        };
        es.onerror = () => {
            es.close();
            setLines([...collected]);
            setLoading(false);
        };
        return () => es.close();
    }, [jobId]);

    // Scroll to bottom when first loaded
    useEffect(() => {
        if (!loading && containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    }, [loading]);

    return (
        <div className='JobLogPanel'>
            <div className='JobLogHeader'>
                <span>Log Output</span>
                <span className='JobLogLines'>{lines.length} lines</span>
            </div>
            <div className='JobLogContainer' ref={containerRef}>
                {lines.map((line, i) => (
                    <div className='JobLogLine' key={i}>{line}</div>
                ))}
                {loading && lines.length === 0 && (
                    <div className='JobLogPlaceholder'>Loading log...</div>
                )}
                {!loading && lines.length === 0 && (
                    <div className='JobLogPlaceholder'>No log output</div>
                )}
            </div>
        </div>
    );
};

const JobDashboard: React.FC = () => {
    const [jobs, setJobs] = useState<JobInfo[]>([]);
    const [expanded, setExpanded] = useState(true);
    const [selectedJob, setSelectedJob] = useState<string | null>(null);
    const [cancelling, setCancelling] = useState<Set<string>>(new Set());

    const fetchJobs = useCallback(async () => {
        try {
            const res = await fetch(`${API_URL}/api/jobs`);
            if (!res.ok) return;
            const data: JobInfo[] = await res.json();
            data.sort((a, b) => {
                const sa = STATUS_ORDER[a.status] ?? 5;
                const sb = STATUS_ORDER[b.status] ?? 5;
                if (sa !== sb) return sa - sb;
                return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
            });
            setJobs(data);
        } catch { /* silent */ }
    }, []);

    useEffect(() => {
        fetchJobs();
        const timer = setInterval(fetchJobs, POLL_INTERVAL);
        return () => clearInterval(timer);
    }, [fetchJobs]);

    const handleCancel = async (jobId: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setCancelling(prev => new Set(prev).add(jobId));
        try {
            await fetch(`${API_URL}/api/jobs/${jobId}/cancel`, { method: 'POST' });
            setTimeout(fetchJobs, 500);
        } catch { /* silent */ }
        setCancelling(prev => {
            const next = new Set(prev);
            next.delete(jobId);
            return next;
        });
    };

    const toggleJobDetail = (jobId: string) => {
        setSelectedJob(prev => prev === jobId ? null : jobId);
    };

    const running = jobs.filter(j => j.status === 'running' || j.status === 'pending');
    const recent = jobs.filter(j => j.status !== 'running' && j.status !== 'pending').slice(0, 15);
    const selected = jobs.find(j => j.job_id === selectedJob) || null;

    const renderJobRow = (job: JobInfo, showCancel: boolean) => {
        const isSelected = selectedJob === job.job_id;
        return (
            <div key={job.job_id}>
                <div
                    className={`JobRow status-${job.status} ${isSelected ? 'selected' : ''}`}
                    onClick={() => toggleJobDetail(job.job_id)}
                    role='button'
                    tabIndex={0}
                    aria-expanded={isSelected}
                    onKeyDown={(e) => { if (e.key === 'Enter') toggleJobDetail(job.job_id); }}
                >
                    <span className={`JobStatusDot ${job.status}`} />
                    <span className='JobType'>{job.job_type}</span>
                    <code className='JobId'>{job.job_id}</code>
                    {!showCancel && <span className='JobTime'>{relativeTime(job.created_at)}</span>}
                    <span className='JobDuration'>{duration(job.started_at, job.finished_at)}</span>
                    {showCancel ? (
                        <button
                            type='button'
                            className='JobCancelBtn'
                            onClick={(e) => handleCancel(job.job_id, e)}
                            disabled={cancelling.has(job.job_id)}
                            aria-label={`Cancel job ${job.job_id}`}
                        >
                            {cancelling.has(job.job_id) ? '...' : 'Cancel'}
                        </button>
                    ) : (
                        <span className={`JobStatusLabel ${job.status}`}>{job.status}</span>
                    )}
                </div>
                {isSelected && selected && (
                    <div className='JobDetail'>
                        <div className='JobDetailGrid'>
                            <div className='JobDetailField'>
                                <span className='JobDetailKey'>Job ID</span>
                                <code className='JobDetailValue selectable'>{selected.job_id}</code>
                            </div>
                            <div className='JobDetailField'>
                                <span className='JobDetailKey'>Type</span>
                                <span className='JobDetailValue'>{selected.job_type}</span>
                            </div>
                            <div className='JobDetailField'>
                                <span className='JobDetailKey'>Status</span>
                                <span className={`JobDetailValue status-text-${selected.status}`}>{selected.status}</span>
                            </div>
                            <div className='JobDetailField'>
                                <span className='JobDetailKey'>Created</span>
                                <span className='JobDetailValue'>{formatTime(selected.created_at)}</span>
                            </div>
                            <div className='JobDetailField'>
                                <span className='JobDetailKey'>Started</span>
                                <span className='JobDetailValue'>{formatTime(selected.started_at)}</span>
                            </div>
                            <div className='JobDetailField'>
                                <span className='JobDetailKey'>Finished</span>
                                <span className='JobDetailValue'>{formatTime(selected.finished_at)}</span>
                            </div>
                            <div className='JobDetailField'>
                                <span className='JobDetailKey'>Duration</span>
                                <span className='JobDetailValue'>{duration(selected.started_at, selected.finished_at)}</span>
                            </div>
                            {selected.pid && (
                                <div className='JobDetailField'>
                                    <span className='JobDetailKey'>PID</span>
                                    <span className='JobDetailValue'>{selected.pid}</span>
                                </div>
                            )}
                            {selected.error_message && (
                                <div className='JobDetailField full'>
                                    <span className='JobDetailKey'>Error</span>
                                    <span className='JobDetailValue error'>{selected.error_message}</span>
                                </div>
                            )}
                            {selected.args && Object.keys(selected.args).length > 0 && (
                                <div className='JobDetailField full'>
                                    <span className='JobDetailKey'>Args</span>
                                    <div className='JobDetailArgs'>
                                        {Object.entries(selected.args).map(([k, v]) => (
                                            <div key={k} className='ArgRow'>
                                                <span className='ArgKey'>{k}</span>
                                                <span className='ArgValue'>{String(v)}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                        <JobLogPanel jobId={selected.job_id} />
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className='JobDashboard'>
            <button
                type='button'
                className='DashboardToggle'
                onClick={() => setExpanded(!expanded)}
                aria-expanded={expanded}
                aria-label='Toggle job dashboard'
            >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className={`ToggleChevron ${expanded ? 'open' : ''}`}>
                    <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                <span className='DashboardTitle'>Jobs</span>
                {running.length > 0 && (
                    <span className='ActiveCount'>{running.length} active</span>
                )}
                <span className='TotalCount'>{jobs.length} total</span>
            </button>

            {expanded && (
                <div className='DashboardBody'>
                    {jobs.length === 0 && (
                        <div className='DashboardEmpty'>No jobs yet. Run a pipeline to see jobs here.</div>
                    )}

                    {running.length > 0 && (
                        <div className='JobSection'>
                            <div className='JobSectionLabel'>Active</div>
                            {running.map(job => renderJobRow(job, true))}
                        </div>
                    )}

                    {recent.length > 0 && (
                        <div className='JobSection'>
                            <div className='JobSectionLabel'>Recent</div>
                            {recent.map(job => renderJobRow(job, false))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default JobDashboard;
