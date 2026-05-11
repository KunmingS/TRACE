import React, { useEffect, useMemo, useState } from 'react';
import { connect } from 'react-redux';
import { Scrollbars } from 'react-custom-scrollbars-2';
import { saveAs } from 'file-saver';
import { GenericYesNoPopup } from '../GenericYesNoPopup/GenericYesNoPopup';
import { AppState } from '../../../store';
import { updateActivePopupType } from '../../../store/general/actionCreators';
import { submitNewNotification } from '../../../store/notifications/actionCreators';
import { NotificationUtil } from '../../../utils/NotificationUtil';
import { PopupWindowType } from '../../../data/enums/PopupWindowType';
import { CSVImporter, CSVRow } from '../../../logic/import/csv/CSVImporter';
import { Settings } from '../../../settings/Settings';
import { API_URL } from '../../../config';
import './AnnotationPreviewPopup.scss';

interface PreviewPayload {
    file: string;
    csvName: string;
    dir: string;
}

interface IProps {
    payload: PreviewPayload | null;
    updateActivePopupTypeAction: (type: PopupWindowType) => void;
    submitNewNotification: typeof submitNewNotification;
}

type LoadState =
    | { kind: 'loading' }
    | { kind: 'error'; message: string }
    | { kind: 'ready'; rows: CSVRow[]; traceMetaLine: string | null };

const formatFrames = (n: number | undefined): string =>
    n == null || !Number.isFinite(n) ? '—' : n.toLocaleString('en-US');

// Stable color picker: hash a behavior name into the palette so two runs
// of the same CSV always show the same dots, even though the CSV doesn't
// store explicit colors. The behavior popup uses the palette directly;
// here we just need a deterministic mapping.
const colorForBehavior = (name: string): string => {
    const palette = Settings.LABEL_COLORS_PALETTE;
    let hash = 0;
    for (let i = 0; i < name.length; i++) {
        hash = ((hash << 5) - hash + name.charCodeAt(i)) | 0;
    }
    return palette[Math.abs(hash) % palette.length];
};

const AnnotationPreviewPopup: React.FC<IProps> = ({ payload, updateActivePopupTypeAction, submitNewNotification }) => {
    const [state, setState] = useState<LoadState>({ kind: 'loading' });

    useEffect(() => {
        if (!payload) return undefined;
        let cancelled = false;
        const fetchCsv = async () => {
            try {
                const params = new URLSearchParams({
                    dir: payload.dir,
                    csvName: payload.csvName,
                });
                const res = await fetch(
                    `${API_URL}/api/files/${encodeURIComponent(payload.file)}/csv?${params}`,
                );
                if (!res.ok) {
                    if (cancelled) return;
                    if (res.status === 404) {
                        setState({ kind: 'ready', rows: [], traceMetaLine: null });
                        return;
                    }
                    setState({ kind: 'error', message: `Server returned ${res.status}` });
                    return;
                }
                const text = await res.text();
                const { rows } = CSVImporter.parseCSVWithMeta(text);
                const traceMetaLine = text
                    .split(/\r?\n/)
                    .find(line => /^#\s*trace-meta:/i.test(line.trim()))
                    ?.trim() ?? null;
                if (cancelled) return;
                setState({ kind: 'ready', rows, traceMetaLine });
            } catch (e: any) {
                if (cancelled) return;
                setState({ kind: 'error', message: e?.message || 'Failed to load CSV' });
            }
        };
        setState({ kind: 'loading' });
        void fetchCsv();
        return () => { cancelled = true; };
    }, [payload]);

    const stats = useMemo(() => {
        if (state.kind !== 'ready') return { count: 0, behaviors: 0, subjects: 0 };
        const behaviors = new Set<string>();
        const subjects = new Set<string>();
        for (const r of state.rows) {
            if (r.behavior) behaviors.add(r.behavior);
            if (r.animal) subjects.add(r.animal);
        }
        return { count: state.rows.length, behaviors: behaviors.size, subjects: subjects.size };
    }, [state]);

    const onClose = () => updateActivePopupTypeAction(null);

    // Re-fetch and save to disk. Mirrors the FileBrowser download path so
    // the user can grab the CSV without leaving the preview.
    const onDownload = async () => {
        if (!payload) return;
        try {
            const params = new URLSearchParams({ dir: payload.dir, csvName: payload.csvName });
            const res = await fetch(
                `${API_URL}/api/files/${encodeURIComponent(payload.file)}/csv?${params}`,
            );
            if (!res.ok) {
                submitNewNotification(NotificationUtil.createMessageNotification({
                    header: 'Download failed',
                    description: res.status === 404
                        ? `${payload.csvName} was not found on the server.`
                        : `Server returned ${res.status}`,
                }));
                return;
            }
            const text = await res.text();
            const blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
            saveAs(blob, payload.csvName);
        } catch {
            submitNewNotification(NotificationUtil.createMessageNotification({
                header: 'Download failed',
                description: 'Network error while downloading CSV',
            }));
        }
    };

    const renderContent = () => {
        if (!payload) {
            return <div className='AnnotationPreviewPopup empty'>No annotation file selected.</div>;
        }
        return (
            <div className='AnnotationPreviewPopup'>
                {/* Header strip — file name + tally stats. Mirrors the
                    metadata chip pattern used elsewhere in the editor. */}
                <div className='PreviewHead'>
                    <div className='PreviewIdentity'>
                        <span className='PreviewLabel'>Annotation file</span>
                        <span className='PreviewName' title={payload.csvName}>{payload.csvName}</span>
                    </div>
                    <div className='PreviewStats'>
                        <Stat n={stats.count}     label='clips' />
                        <Stat n={stats.behaviors} label='behaviors' />
                        <Stat n={stats.subjects}  label='subjects' />
                        <button
                            type='button'
                            className='DownloadBtn'
                            onClick={() => void onDownload()}
                            disabled={state.kind !== 'ready'}
                            title='Download CSV'
                            aria-label={`Download ${payload.csvName}`}
                        >
                            <svg width='14' height='14' viewBox='0 0 16 16' fill='none' aria-hidden='true'>
                                <path d='M8 2v8m0 0l-3-3m3 3l3-3' stroke='currentColor' strokeWidth='1.4' strokeLinecap='round' strokeLinejoin='round'/>
                                <path d='M3 12v1.5A0.5 0.5 0 0 0 3.5 14h9a0.5 0.5 0 0 0 0.5-0.5V12' stroke='currentColor' strokeWidth='1.4' strokeLinecap='round' strokeLinejoin='round'/>
                            </svg>
                            <span>Download</span>
                        </button>
                    </div>
                </div>

                {state.kind === 'loading' && (
                    <div className='PreviewBody loading'>
                        <span className='LoaderDot' />
                        <span className='LoaderDot' />
                        <span className='LoaderDot' />
                    </div>
                )}

                {state.kind === 'error' && (
                    <div className='PreviewBody error'>
                        <div className='ErrorTitle'>Could not load CSV</div>
                        <div className='ErrorDetail'>{state.message}</div>
                    </div>
                )}

                {state.kind === 'ready' && state.rows.length === 0 && (
                    <div className='PreviewBody empty'>
                        <div className='EmptyMark' aria-hidden='true'>
                            <svg width='32' height='32' viewBox='0 0 32 32' fill='none'>
                                <rect x='6' y='4' width='20' height='24' rx='2' stroke='currentColor' strokeWidth='1.2' strokeDasharray='3 3' fill='none' />
                                <path d='M11 12h10M11 16h10M11 20h6' stroke='currentColor' strokeWidth='1.2' strokeLinecap='round' />
                            </svg>
                        </div>
                        <div className='EmptyTitle'>No annotations yet</div>
                        <div className='EmptyDetail'>Once you label clips, they'll show up here.</div>
                    </div>
                )}

                {state.kind === 'ready' && state.rows.length > 0 && (
                    <div className='PreviewBody table'>
                        {state.traceMetaLine && (
                            <div className='MetaRow' title={state.traceMetaLine}>
                                {state.traceMetaLine}
                            </div>
                        )}
                        <div className='TableHead'>
                            <div className='HeadCell behavior'>Behavior</div>
                            <div className='HeadCell subject'>Subject</div>
                            <div className='HeadCell num'>Start</div>
                            <div className='HeadCell num'>End</div>
                            <div className='HeadCell num last'>Duration</div>
                        </div>
                        <Scrollbars
                            autoHeight
                            autoHeightMin={120}
                            autoHeightMax='var(--annotation-preview-table-max-height)'
                            renderTrackVertical={p => <div {...p} className='ScrollTrackV' />}
                            renderThumbVertical={p => <div {...p} className='ScrollThumb' />}
                        >
                            <div className='TableBody'>
                                {state.rows.map((row, i) => {
                                    const isFrameBased = row.startFrame != null;
                                    const start = isFrameBased ? row.startFrame : row.timestamp;
                                    const end = isFrameBased ? row.endFrame : row.endTimestamp;
                                    const duration = (start != null && end != null)
                                        ? (isFrameBased ? (end as number) - (start as number) + 1 : (end as number) - (start as number))
                                        : null;
                                    return (
                                        <div className='Row' key={i}>
                                            <div className='Cell behavior'>
                                                <span className='Swatch' style={{ background: colorForBehavior(row.behavior) }} />
                                                <span className='BehaviorName' title={row.behavior}>{row.behavior || '—'}</span>
                                            </div>
                                            <div className='Cell subject'>{row.animal || <span className='Muted'>—</span>}</div>
                                            <div className='Cell num'>{formatFrames(start as number)}</div>
                                            <div className='Cell num'>{formatFrames(end as number)}</div>
                                            <div className='Cell num last'>
                                                {duration != null ? formatFrames(duration) : <span className='Muted'>—</span>}
                                                <span className='Unit'>{isFrameBased ? 'fr' : 's'}</span>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </Scrollbars>
                    </div>
                )}
            </div>
        );
    };

    return (
        <GenericYesNoPopup
            title='Annotation preview'
            renderContent={renderContent}
            acceptLabel='Done'
            onAccept={onClose}
            skipRejectButton
        />
    );
};

interface StatProps { n: number; label: string; }
const Stat: React.FC<StatProps> = ({ n, label }) => (
    <span className='StatChip'>
        <span className='StatNum'>{n}</span>
        <span className='StatLabel'>{label}</span>
    </span>
);

const mapStateToProps = (state: AppState) => ({
    payload: state.general.popupPayload as PreviewPayload | null,
});

const mapDispatchToProps = {
    updateActivePopupTypeAction: updateActivePopupType,
    submitNewNotification,
};

export default connect(mapStateToProps, mapDispatchToProps)(AnnotationPreviewPopup);
