import React, { useEffect, useRef, useState } from 'react';
import './LogViewer.scss';
import { API_URL } from '../../../config';

interface LogViewerProps {
    jobId: string | null;
    onCancel?: () => void;
}

// Strip ANSI escape codes for clean display
const stripAnsi = (str: string) => str.replace(/\x1b\[[0-9;]*m/g, '');

const LogViewer: React.FC<LogViewerProps> = ({ jobId, onCancel }) => {
    const [lines, setLines] = useState<string[]>([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [jobStatus, setJobStatus] = useState<string>('');
    const containerRef = useRef<HTMLDivElement>(null);
    const autoScrollRef = useRef(true);

    useEffect(() => {
        if (!jobId) {
            setLines([]);
            setIsStreaming(false);
            setJobStatus('');
            return undefined;
        }

        setLines([]);
        setIsStreaming(true);
        setJobStatus('running');

        const eventSource = new EventSource(`${API_URL}/api/jobs/${jobId}/logs/stream`);

        eventSource.onmessage = (event) => {
            const line = stripAnsi(event.data);
            setLines((prev) => [...prev, line]);
        };

        eventSource.onerror = () => {
            eventSource.close();
            setIsStreaming(false);
            // Fetch final job status
            fetch(`${API_URL}/api/jobs/${jobId}`)
                .then((res) => res.json())
                .then((job) => setJobStatus(job.status || 'unknown'))
                .catch(() => setJobStatus('unknown'));
        };

        return () => {
            eventSource.close();
        };
    }, [jobId]);

    // Auto-scroll to bottom
    useEffect(() => {
        if (autoScrollRef.current && containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    }, [lines]);

    const handleScroll = () => {
        if (!containerRef.current) return;
        const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
        autoScrollRef.current = scrollHeight - scrollTop - clientHeight < 40;
    };

    const handleCancel = async () => {
        if (!jobId) return;
        try {
            await fetch(`${API_URL}/api/jobs/${jobId}/cancel`, { method: 'POST' });
        } catch {}
        onCancel?.();
    };

    if (!jobId) return null;

    const statusClass = jobStatus === 'completed' ? 'success' : jobStatus === 'failed' ? 'error' : '';

    return (
        <div className='LogViewer'>
            <div className='LogHeader'>
                <div className='LogTitle'>
                    {isStreaming && <span className='LiveDot' />}
                    <span>Job Log</span>
                    {jobStatus && !isStreaming && (
                        <span className={`StatusBadge ${statusClass}`}>{jobStatus}</span>
                    )}
                </div>
                <div className='LogActions'>
                    {isStreaming && (
                        <button className='CancelBtn' onClick={handleCancel} type='button'>
                            Cancel
                        </button>
                    )}
                </div>
            </div>
            <div
                className='LogContainer'
                ref={containerRef}
                onScroll={handleScroll}
            >
                {lines.map((line, i) => (
                    <div className='LogLine' key={i}>{line}</div>
                ))}
                {lines.length === 0 && isStreaming && (
                    <div className='LogPlaceholder'>Waiting for output...</div>
                )}
            </div>
        </div>
    );
};

export default LogViewer;
