import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { API_URL } from '../../../config';

interface JobInfo {
    job_id: string;
    job_type: string;
    run_id?: string | null;
    stage?: string | null;
    run_steps?: string[] | null;
    status: string;
    config_path: string | null;
    created_at: string;
    started_at: string | null;
    finished_at: string | null;
    pid: number | null;
    log_file: string | null;
    work_dir: string | null;
    error_message: string | null;
    args?: Record<string, any>;
}

const HIDDEN_ARG_KEYS = new Set([
    'dataset_dir',
    'annotation_path',
    'annotated_video',
    'auto_tune',
    'class_map',
    'disable_deterministic',
    'included_stems',
    'input',
    'not_eval',
    'nproc',
    'output_dir',
    'profile',
    'seed',
    'cfg_options',
]);

function formatArgValue(value: unknown): string {
    if (Array.isArray(value)) return value.join('\n');
    if (value !== null && typeof value === 'object') return JSON.stringify(value, null, 2);
    return String(value);
}

const POLL_ACTIVE_MS = 4000;
const POLL_IDLE_MS = 15000;

type RunStep = 'train' | 'test' | 'infer';
type PipelineSteps = Record<RunStep, boolean>;
type RenderProgress = {
    videoName: string;
    percent: number;
    detail: string;
    complete: boolean;
};

const RUN_STEP_ORDER: RunStep[] = ['train', 'test', 'infer'];
const TASK_ORDER: Record<string, number> = { prep: 0, train: 1, test: 2, infer: 3, 'train-tune': 4 };

function emptySteps(): PipelineSteps {
    return { train: false, test: false, infer: false };
}

function stageForJob(job: JobInfo): string {
    return job.stage || job.job_type;
}

function displayStage(job: JobInfo): string {
    const stage = stageForJob(job);
    return stage === 'infer' ? 'predict' : stage;
}

const isActive = (status: string) => status === 'running' || status === 'pending' || status === 'queued';

function isInferJob(job: JobInfo): boolean {
    return stageForJob(job) === 'infer' || job.job_type === 'infer';
}

function stringList(value: unknown): string[] {
    if (Array.isArray(value)) {
        return value
            .map(item => String(item).trim())
            .filter(Boolean);
    }
    if (typeof value === 'string') {
        return value
            .split(/[\n,]/)
            .map(item => item.trim())
            .filter(Boolean);
    }
    return [];
}

function joinPath(base: string, filename: string): string {
    return `${base.replace(/\/+$/, '')}/${filename}`;
}

function inferVideosLabel(job: JobInfo): string {
    const stems = stringList(job.args?.included_stems);
    return stems.length > 0 ? stems.join('\n') : 'All videos in input folder';
}

function annotatedDestinations(job: JobInfo): string {
    const enabled = Boolean(job.args?.annotated_video);
    if (!enabled) return 'Off';

    const outputDir = job.work_dir || job.args?.output_dir;
    if (typeof outputDir !== 'string' || !outputDir.trim()) return 'On';

    const stems = stringList(job.args?.included_stems);
    if (stems.length === 0) {
        return joinPath(outputDir, '*_annotated.mp4');
    }
    return stems
        .map(stem => joinPath(outputDir, `${stem}_annotated.mp4`))
        .join('\n');
}

function clampPercent(value: number): number {
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(100, value));
}

function formatPercent(value: number): string {
    const clamped = clampPercent(value);
    return `${clamped >= 10 || clamped === 0 ? clamped.toFixed(0) : clamped.toFixed(1)}%`;
}

function isAnnotatedRenderLine(line: string): boolean {
    return (
        line.includes('Annotated render starts:')
        || line.includes('Annotated render progress:')
        || line.includes('Annotated render complete:')
    );
}

function parseAnnotatedRenderProgress(lines: string[]): RenderProgress | null {
    let progress: RenderProgress | null = null;

    for (const line of lines) {
        const start = line.match(/Annotated render starts:\s+(.+?)\s+\((\d+) frames,/);
        if (start) {
            const total = Number(start[2]);
            progress = {
                videoName: start[1],
                percent: 0,
                detail: `0/${total.toLocaleString()} frames`,
                complete: false,
            };
            continue;
        }

        const update = line.match(
            /Annotated render progress:\s+(.+?)\s+(\d+)\/(\d+) frames \((\d+(?:\.\d+)?)%([^)]*)\)/,
        );
        if (update) {
            const current = Number(update[2]);
            const total = Number(update[3]);
            const extra = update[5].replace(/^,\s*/, '').trim();
            progress = {
                videoName: update[1],
                percent: clampPercent(Number(update[4])),
                detail: `${current.toLocaleString()}/${total.toLocaleString()} frames${extra ? `, ${extra}` : ''}`,
                complete: false,
            };
            continue;
        }

        const complete = line.match(/Annotated render complete:\s+(.+?)\s+\((\d+)\/(\d+) frames\)/);
        if (complete) {
            const current = Number(complete[2]);
            const total = Number(complete[3]);
            progress = {
                videoName: complete[1],
                percent: 100,
                detail: `${current.toLocaleString()}/${total.toLocaleString()} frames`,
                complete: true,
            };
        }
    }

    return progress;
}

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

/* Aggregate task statuses into a single pipeline status.
 * Priority: running > pending > failed > cancelled > completed > empty.
 */
function aggregatePipelineStatus(tasks: JobInfo[]): string {
    if (tasks.length === 0) return 'pending';
    if (tasks.some(t => t.status === 'running')) return 'running';
    if (tasks.some(t => t.status === 'pending' || t.status === 'queued')) return 'pending';
    if (tasks.some(t => t.status === 'failed')) return 'failed';
    if (tasks.some(t => t.status === 'cancelled')) return 'cancelled';
    if (tasks.every(t => t.status === 'completed')) return 'completed';
    return tasks[0].status;
}

function pipelineSpan(tasks: JobInfo[]): { start: string | null; end: string | null } {
    let start: number | null = null;
    let end: number | null = null;
    let anyActive = false;
    for (const t of tasks) {
        if (t.started_at) {
            const s = new Date(t.started_at).getTime();
            if (start === null || s < start) start = s;
        }
        if (isActive(t.status)) {
            anyActive = true;
        } else if (t.finished_at) {
            const e = new Date(t.finished_at).getTime();
            if (end === null || e > end) end = e;
        }
    }
    return {
        start: start !== null ? new Date(start).toISOString() : null,
        end: anyActive ? null : (end !== null ? new Date(end).toISOString() : null),
    };
}

interface Pipeline {
    id: string;
    createdAt: string;
    tasks: JobInfo[];
    steps: PipelineSteps;
}

function stepsFromJobs(tasks: JobInfo[]): PipelineSteps {
    const steps = emptySteps();
    for (const job of tasks) {
        if (Array.isArray(job.run_steps)) {
            for (const step of job.run_steps) {
                if (step === 'train' || step === 'test' || step === 'infer') steps[step] = true;
            }
        }
        const stage = stageForJob(job);
        if (stage === 'train' || stage === 'test' || stage === 'infer') steps[stage] = true;
    }
    if (!steps.train && !steps.test && !steps.infer && tasks.some(job => stageForJob(job) === 'prep')) {
        steps.train = true;
    }
    return steps;
}

function firstCreatedAt(tasks: JobInfo[], fallback: string): string {
    let created = new Date(fallback).getTime();
    for (const task of tasks) {
        const t = new Date(task.created_at).getTime();
        if (t < created) created = t;
    }
    return new Date(created).toISOString();
}

/* ── Job Log Viewer (reads stored logs for past jobs) ── */
const JobLogPanel: React.FC<{ jobId: string }> = ({ jobId }) => {
    const [lines, setLines] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);
    const containerRef = useRef<HTMLDivElement>(null);
    const renderProgress = useMemo(() => parseAnnotatedRenderProgress(lines), [lines]);

    useEffect(() => {
        setLines([]);
        setLoading(true);
        const collected: string[] = [];
        const es = new EventSource(`${API_URL}/api/jobs/${jobId}/logs/stream`);

        es.onmessage = (event) => {
            const cleanLine = event.data.replace(/\x1b\[[0-9;]*m/g, '');
            collected.push(cleanLine);
            if (collected.length % 20 === 0 || collected.length < 5 || isAnnotatedRenderLine(cleanLine)) {
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
            {renderProgress && (
                <div className={`JobProgress ${renderProgress.complete ? 'complete' : 'running'}`}>
                    <div className='JobProgressMeta'>
                        <span className='JobProgressTitle'>
                            {renderProgress.complete ? 'Annotated video ready' : 'Rendering annotated video'}
                        </span>
                        <span className='JobProgressPercent'>{formatPercent(renderProgress.percent)}</span>
                    </div>
                    <div
                        className='JobProgressTrack'
                        role='progressbar'
                        aria-label='Annotated video render progress'
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={Math.round(clampPercent(renderProgress.percent))}
                    >
                        <div
                            className='JobProgressFill'
                            style={{ width: `${clampPercent(renderProgress.percent)}%` }}
                        />
                    </div>
                    <div className='JobProgressDetail'>
                        <span className='JobProgressVideo'>{renderProgress.videoName}</span>
                        <span>{renderProgress.detail}</span>
                    </div>
                </div>
            )}
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

/* Running pill + Cancel button rendered together.
 * Replaces the old standalone red Cancel button so active work doesn't look like an error.
 */
const RunningControls: React.FC<{
    status: string;
    onCancel: (e: React.MouseEvent) => void;
    cancelling: boolean;
}> = ({ status, onCancel, cancelling }) => (
    <span className='RunningControls' onClick={(e) => e.stopPropagation()}>
        <span className={`JobStatusLabel ${status}`}>
            <span className='RunningDot' />
            {status === 'pending' || status === 'queued' ? 'Queued' : 'Running'}
        </span>
        <button
            type='button'
            className='JobCancelBtn'
            onClick={onCancel}
            disabled={cancelling}
            aria-label='Cancel'
        >
            {cancelling ? 'Cancelling...' : 'Cancel'}
        </button>
    </span>
);

const DeleteButton: React.FC<{
    onDelete: (e: React.MouseEvent) => void;
    deleting: boolean;
    label: string;
}> = ({ onDelete, deleting, label }) => (
    <button
        type='button'
        className='JobDeleteBtn'
        onClick={(e) => { e.stopPropagation(); onDelete(e); }}
        disabled={deleting}
        aria-label={label}
        title={label}
    >
        <svg width='12' height='12' viewBox='0 0 12 12' fill='none' aria-hidden='true'>
            <path d='M3 3l6 6M9 3l-6 6' stroke='currentColor' strokeWidth='1.4' strokeLinecap='round'/>
        </svg>
    </button>
);

const JobDashboard: React.FC = () => {
    const [jobs, setJobs] = useState<JobInfo[]>([]);
    const [expanded, setExpanded] = useState(true);
    const [expandedPipelines, setExpandedPipelines] = useState<Set<string>>(new Set());
    const autoExpandedRef = useRef<Set<string>>(new Set());
    const [selectedJob, setSelectedJob] = useState<string | null>(null);
    const [cancelling, setCancelling] = useState<Set<string>>(new Set());
    const [deleting, setDeleting] = useState<Set<string>>(new Set());

    const fetchJobs = useCallback(async () => {
        try {
            const res = await fetch(`${API_URL}/api/jobs`);
            if (!res.ok) return;
            const data: JobInfo[] = await res.json();
            setJobs(data);
        } catch { /* silent */ }
    }, []);

    const hasRunningJobs = jobs.some(j => isActive(j.status));

    useEffect(() => {
        let cancelledFlag = false;
        let timer: ReturnType<typeof setTimeout> | null = null;

        const tick = async () => {
            if (cancelledFlag) return;
            if (typeof document !== 'undefined' && document.hidden) {
                timer = setTimeout(tick, POLL_ACTIVE_MS);
                return;
            }
            await fetchJobs();
            if (cancelledFlag) return;
            const delay = hasRunningJobs ? POLL_ACTIVE_MS : POLL_IDLE_MS;
            timer = setTimeout(tick, delay);
        };

        const onVisible = () => {
            if (!document.hidden) {
                if (timer) clearTimeout(timer);
                tick();
            }
        };
        document.addEventListener('visibilitychange', onVisible);

        tick();
        return () => {
            cancelledFlag = true;
            if (timer) clearTimeout(timer);
            document.removeEventListener('visibilitychange', onVisible);
        };
    }, [fetchJobs, hasRunningJobs]);

    /* ── Group jobs into backend-authored pipeline runs ── */
    const { pipelines, orphans } = useMemo(() => {
        const pipelineMap = new Map<string, Pipeline>();
        const orphans: JobInfo[] = [];

        for (const job of jobs) {
            const runId = job.run_id;
            if (!runId) {
                orphans.push(job);
                continue;
            }

            let pipeline = pipelineMap.get(runId);
            if (!pipeline) {
                pipeline = {
                    id: runId,
                    createdAt: job.created_at,
                    steps: emptySteps(),
                    tasks: [],
                };
                pipelineMap.set(runId, pipeline);
            }
            pipeline.tasks.push(job);
        }

        const pipelineList = Array.from(pipelineMap.values());

        for (const p of pipelineList) {
            p.steps = stepsFromJobs(p.tasks);
            p.createdAt = firstCreatedAt(p.tasks, p.createdAt);
            p.tasks.sort((a, b) => {
                const oa = TASK_ORDER[stageForJob(a)] ?? 99;
                const ob = TASK_ORDER[stageForJob(b)] ?? 99;
                if (oa !== ob) return oa - ob;
                return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
            });
        }

        pipelineList.sort((a, b) => {
            const sa = aggregatePipelineStatus(a.tasks);
            const sb = aggregatePipelineStatus(b.tasks);
            const aActive = isActive(sa);
            const bActive = isActive(sb);
            if (aActive !== bActive) return aActive ? -1 : 1;
            return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
        });

        orphans.sort((a, b) => {
            const aActive = isActive(a.status);
            const bActive = isActive(b.status);
            if (aActive !== bActive) return aActive ? -1 : 1;
            return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        });

        return { pipelines: pipelineList, orphans };
    }, [jobs]);

    /* Auto-expand any pipeline that has an active task, but only once per pipeline —
     * once the user manually collapses it, we honour that choice even while it keeps running.
     */
    useEffect(() => {
        setExpandedPipelines(prev => {
            const next = new Set(prev);
            let changed = false;
            for (const p of pipelines) {
                if (isActive(aggregatePipelineStatus(p.tasks)) && !autoExpandedRef.current.has(p.id)) {
                    autoExpandedRef.current.add(p.id);
                    if (!next.has(p.id)) {
                        next.add(p.id);
                        changed = true;
                    }
                }
            }
            return changed ? next : prev;
        });
    }, [pipelines]);

    const cancelJob = async (jobId: string) => {
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

    const cancelPipeline = async (p: Pipeline, e: React.MouseEvent) => {
        e.stopPropagation();
        // Cancel whichever task in this pipeline is active (backend runs one at a time).
        const active = p.tasks.find(t => isActive(t.status));
        if (active) await cancelJob(active.job_id);
    };

    const deleteJob = async (jobId: string) => {
        setDeleting(prev => new Set(prev).add(jobId));
        try {
            await fetch(`${API_URL}/api/jobs/${jobId}`, { method: 'DELETE' });
        } catch { /* silent */ }
        setDeleting(prev => {
            const next = new Set(prev);
            next.delete(jobId);
            return next;
        });
        await fetchJobs();
    };

    const deletePipeline = async (p: Pipeline, e: React.MouseEvent) => {
        e.stopPropagation();
        // Backend only blocks deletes for active jobs; we already gate the button on
        // the aggregate status, so every task here is safe to delete.
        const ids = p.tasks.map(t => t.job_id);
        setDeleting(prev => {
            const next = new Set(prev);
            ids.forEach(id => next.add(id));
            return next;
        });
        try {
            await Promise.all(
                ids.map(id => fetch(`${API_URL}/api/jobs/${id}`, { method: 'DELETE' }))
            );
        } catch { /* silent */ }
        setDeleting(prev => {
            const next = new Set(prev);
            ids.forEach(id => next.delete(id));
            return next;
        });
        await fetchJobs();
    };

    const togglePipeline = (pid: string) => {
        setExpandedPipelines(prev => {
            const next = new Set(prev);
            if (next.has(pid)) next.delete(pid); else next.add(pid);
            return next;
        });
    };

    const toggleJobDetail = (jobId: string, e?: React.MouseEvent) => {
        if (e) e.stopPropagation();
        setSelectedJob(prev => prev === jobId ? null : jobId);
    };

    const activePipelines = pipelines.filter(p => isActive(aggregatePipelineStatus(p.tasks)));
    const recentPipelines = pipelines.filter(p => !isActive(aggregatePipelineStatus(p.tasks))).slice(0, 10);
    const activeOrphans = orphans.filter(j => isActive(j.status));
    const recentOrphans = orphans.filter(j => !isActive(j.status)).slice(0, 10);

    const stepsLabel = (steps: PipelineSteps): string => {
        const parts = RUN_STEP_ORDER
            .filter(step => steps[step])
            .map(step => step === 'infer' ? 'Predict' : step[0].toUpperCase() + step.slice(1));
        return parts.join(' → ') || 'Run Model';
    };

    const renderTaskRow = (job: JobInfo) => {
        const isSelected = selectedJob === job.job_id;
        const active = isActive(job.status);
        return (
            <div key={job.job_id} className='TaskRowWrap'>
                <div
                    className={`JobRow TaskRow status-${job.status} ${isSelected ? 'selected' : ''}`}
                    onClick={(e) => toggleJobDetail(job.job_id, e)}
                    role='button'
                    tabIndex={0}
                    aria-expanded={isSelected}
                    onKeyDown={(e) => { if (e.key === 'Enter') toggleJobDetail(job.job_id); }}
                >
                    <span className={`JobStatusDot ${job.status}`} />
                    <span className='JobType'>{displayStage(job)}</span>
                    <code className='JobId'>{job.job_id.slice(0, 10)}</code>
                    <span className='JobDuration'>{duration(job.started_at, job.finished_at)}</span>
                    {active ? (
                        <RunningControls
                            status={job.status}
                            cancelling={cancelling.has(job.job_id)}
                            onCancel={(e) => { e.stopPropagation(); cancelJob(job.job_id); }}
                        />
                    ) : (
                        <>
                            <span className={`JobStatusLabel ${job.status}`}>{job.status}</span>
                            <DeleteButton
                                deleting={deleting.has(job.job_id)}
                                onDelete={() => deleteJob(job.job_id)}
                                label='Delete job'
                            />
                        </>
                    )}
                </div>
                {isSelected && <JobDetailPanel job={job} />}
            </div>
        );
    };

    const renderOrphanRow = (job: JobInfo) => {
        const isSelected = selectedJob === job.job_id;
        const active = isActive(job.status);
        return (
            <div key={job.job_id} className='OrphanRowWrap'>
                <div
                    className={`JobRow OrphanRow status-${job.status} ${isSelected ? 'selected' : ''}`}
                    onClick={(e) => toggleJobDetail(job.job_id, e)}
                    role='button'
                    tabIndex={0}
                    aria-expanded={isSelected}
                    onKeyDown={(e) => { if (e.key === 'Enter') toggleJobDetail(job.job_id); }}
                >
                    <span className={`JobStatusDot ${job.status}`} />
                    <span className='JobType'>{displayStage(job)}</span>
                    <code className='JobId'>{job.job_id.slice(0, 10)}</code>
                    {!active && <span className='JobTime'>{relativeTime(job.created_at)}</span>}
                    <span className='JobDuration'>{duration(job.started_at, job.finished_at)}</span>
                    {active ? (
                        <RunningControls
                            status={job.status}
                            cancelling={cancelling.has(job.job_id)}
                            onCancel={(e) => { e.stopPropagation(); cancelJob(job.job_id); }}
                        />
                    ) : (
                        <>
                            <span className={`JobStatusLabel ${job.status}`}>{job.status}</span>
                            <DeleteButton
                                deleting={deleting.has(job.job_id)}
                                onDelete={() => deleteJob(job.job_id)}
                                label='Delete job'
                            />
                        </>
                    )}
                </div>
                {isSelected && <JobDetailPanel job={job} />}
            </div>
        );
    };

    const renderPipelineRow = (p: Pipeline) => {
        const status = aggregatePipelineStatus(p.tasks);
        const span = pipelineSpan(p.tasks);
        const active = isActive(status);
        const isOpen = expandedPipelines.has(p.id);
        return (
            <div key={p.id} className='PipelineGroupWrap'>
                <div
                    className={`JobRow PipelineRow status-${status} ${isOpen ? 'open' : ''}`}
                    onClick={() => togglePipeline(p.id)}
                    role='button'
                    tabIndex={0}
                    aria-expanded={isOpen}
                    onKeyDown={(e) => { if (e.key === 'Enter') togglePipeline(p.id); }}
                >
                    <svg width='10' height='10' viewBox='0 0 12 12' fill='none' className={`PipelineChevron ${isOpen ? 'open' : ''}`}>
                        <path d='M3 4.5L6 7.5L9 4.5' stroke='currentColor' strokeWidth='1.3' strokeLinecap='round' strokeLinejoin='round'/>
                    </svg>
                    <span className={`JobStatusDot ${status}`} />
                    <span className='PipelineLabel'>{stepsLabel(p.steps)}</span>
                    <span className='PipelineTaskCount'>{p.tasks.length} task{p.tasks.length === 1 ? '' : 's'}</span>
                    {!active && <span className='JobTime'>{relativeTime(p.createdAt)}</span>}
                    <span className='JobDuration'>{duration(span.start, span.end)}</span>
                    {active ? (
                        <RunningControls
                            status={status}
                            cancelling={p.tasks.some(t => cancelling.has(t.job_id))}
                            onCancel={(e) => cancelPipeline(p, e)}
                        />
                    ) : (
                        <>
                            <span className={`JobStatusLabel ${status}`}>{status}</span>
                            <DeleteButton
                                deleting={p.tasks.some(t => deleting.has(t.job_id))}
                                onDelete={(e) => deletePipeline(p, e)}
                                label='Delete run and its tasks'
                            />
                        </>
                    )}
                </div>
                {isOpen && (
                    <div className='PipelineTaskList'>
                        {p.tasks.map(t => renderTaskRow(t))}
                    </div>
                )}
            </div>
        );
    };

    const totalCount = pipelines.length + orphans.length;

    return (
        <div className='JobDashboard'>
            <button
                type='button'
                className='DashboardToggle'
                onClick={() => setExpanded(!expanded)}
                aria-expanded={expanded}
                aria-label='Toggle job dashboard'
            >
                <svg width='12' height='12' viewBox='0 0 12 12' fill='none' className={`ToggleChevron ${expanded ? 'open' : ''}`}>
                    <path d='M3 4.5L6 7.5L9 4.5' stroke='currentColor' strokeWidth='1.3' strokeLinecap='round' strokeLinejoin='round'/>
                </svg>
                <span className='DashboardTitle'>Jobs</span>
                {(activePipelines.length + activeOrphans.length) > 0 && (
                    <span className='ActiveCount'>{activePipelines.length + activeOrphans.length} active</span>
                )}
                <span className='TotalCount'>{totalCount} total</span>
            </button>

            {expanded && (
                <div className='DashboardBody'>
                    {totalCount === 0 && (
                        <div className='DashboardEmpty'>No jobs yet. Run a model to see jobs here.</div>
                    )}

                    {(activePipelines.length > 0 || activeOrphans.length > 0) && (
                        <div className='JobSection'>
                            <div className='JobSectionLabel'>Active</div>
                            {activePipelines.map(renderPipelineRow)}
                            {activeOrphans.map(renderOrphanRow)}
                        </div>
                    )}

                    {(recentPipelines.length > 0 || recentOrphans.length > 0) && (
                        <div className='JobSection'>
                            <div className='JobSectionLabel'>Recent</div>
                            {recentPipelines.map(renderPipelineRow)}
                            {recentOrphans.map(renderOrphanRow)}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

const JobDetailPanel: React.FC<{ job: JobInfo }> = ({ job }) => (
    <div className='JobDetail'>
        <div className='JobDetailGrid'>
            <div className='JobDetailField'>
                <span className='JobDetailKey'>Job ID</span>
                <code className='JobDetailValue selectable'>{job.job_id}</code>
            </div>
            {job.run_id && (
                <div className='JobDetailField'>
                    <span className='JobDetailKey'>Run ID</span>
                    <code className='JobDetailValue selectable'>{job.run_id}</code>
                </div>
            )}
            {job.stage && (
                <div className='JobDetailField'>
                    <span className='JobDetailKey'>Stage</span>
                    <span className='JobDetailValue'>{displayStage(job)}</span>
                </div>
            )}
            <div className='JobDetailField'>
                <span className='JobDetailKey'>Type</span>
                <span className='JobDetailValue'>{job.job_type}</span>
            </div>
            <div className='JobDetailField'>
                <span className='JobDetailKey'>Status</span>
                <span className={`JobDetailValue status-text-${job.status}`}>{job.status}</span>
            </div>
            <div className='JobDetailField'>
                <span className='JobDetailKey'>Created</span>
                <span className='JobDetailValue'>{formatTime(job.created_at)}</span>
            </div>
            <div className='JobDetailField'>
                <span className='JobDetailKey'>Started</span>
                <span className='JobDetailValue'>{formatTime(job.started_at)}</span>
            </div>
            <div className='JobDetailField'>
                <span className='JobDetailKey'>Finished</span>
                <span className='JobDetailValue'>{formatTime(job.finished_at)}</span>
            </div>
            <div className='JobDetailField'>
                <span className='JobDetailKey'>Duration</span>
                <span className='JobDetailValue'>{duration(job.started_at, job.finished_at)}</span>
            </div>
            {job.pid && (
                <div className='JobDetailField'>
                    <span className='JobDetailKey'>PID</span>
                    <span className='JobDetailValue'>{job.pid}</span>
                </div>
            )}
            {job.work_dir && (
                <div className='JobDetailField full'>
                    <span className='JobDetailKey'>Work Dir</span>
                    <code className='JobDetailValue selectable'>{job.work_dir}</code>
                </div>
            )}
            {isInferJob(job) && typeof job.args?.input === 'string' && (
                <div className='JobDetailField full'>
                    <span className='JobDetailKey'>Inference Input</span>
                    <code className='JobDetailValue selectable'>{job.args.input}</code>
                </div>
            )}
            {isInferJob(job) && (
                <div className='JobDetailField full'>
                    <span className='JobDetailKey'>Videos</span>
                    <code className='JobDetailValue selectable multiline'>{inferVideosLabel(job)}</code>
                </div>
            )}
            {isInferJob(job) && (
                <div className='JobDetailField full'>
                    <span className='JobDetailKey'>Annotated Video</span>
                    <code className={`JobDetailValue selectable multiline${job.args?.annotated_video ? '' : ' muted'}`}>
                        {annotatedDestinations(job)}
                    </code>
                </div>
            )}
            {job.log_file && (
                <div className='JobDetailField full'>
                    <span className='JobDetailKey'>Log File</span>
                    <code className='JobDetailValue selectable'>{job.log_file}</code>
                </div>
            )}
            {job.error_message && (
                <div className='JobDetailField full'>
                    <span className='JobDetailKey'>Error</span>
                    <span className='JobDetailValue error'>{job.error_message}</span>
                </div>
            )}
            {(() => {
                const visible = Object.entries(job.args || {}).filter(
                    ([k]) => !HIDDEN_ARG_KEYS.has(k),
                );
                if (visible.length === 0) return null;
                return (
                    <div className='JobDetailField full'>
                        <span className='JobDetailKey'>Args</span>
                        <div className='JobDetailArgs'>
                            {visible.map(([k, v]) => (
                                <div key={k} className='ArgRow'>
                                    <span className='ArgKey'>{k}</span>
                                    <span className='ArgValue'>{formatArgValue(v)}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                );
            })()}
        </div>
        <JobLogPanel jobId={job.job_id} />
    </div>
);

export default JobDashboard;
