import React, { useState, useEffect } from 'react';
import { connect } from 'react-redux';
import { AppState } from '../../../store';
import { ImageData, LabelName, Subject } from '../../../store/labels/types';
import { addImageData, updateActiveImageIndex, updateImageData, updateLabelNames, updateSubjects } from '../../../store/labels/actionCreators';
import { clearVideoHasCsv, updateActivePopupType, updatePopupPayload, updateVideoDirectory, updateVideoFiles } from '../../../store/general/actionCreators';
import { PopupWindowType } from '../../../data/enums/PopupWindowType';
import { ImageDataUtil } from '../../../utils/ImageDataUtil';
import { CSVImporter } from '../../../logic/import/csv/CSVImporter';
import { setPendingFocusLabelId } from '../../PopupView/InsertLabelNamesPopup/focusState';
import { API_URL } from '../../../config';
import classNames from 'classnames';
import { submitNewNotification } from '../../../store/notifications/actionCreators';
import { NotificationUtil } from '../../../utils/NotificationUtil';
import { decideServeAction, browserCanPlay, ServeAction } from '../../../utils/CodecSupport';
import { saveAs } from 'file-saver';
import PathPicker from '../../Common/PathPicker/PathPicker';
import Tooltip from '../../Common/Tooltip/Tooltip';
import './FileBrowser.scss';

interface FileInfo {
    name: string;
    hasCsv: boolean;
    // List of CSVs that belong to this video — `{base}.csv` plus any
    // `{base}_*.csv` (rater A vs rater B, draft vs final, …). The
    // canonical `{base}.csv` is sorted first when present. Optional for
    // older backends.
    csvFiles?: string[];
    codec?: string;
    container?: string;
    isH264?: boolean;
    hasCachedH264?: boolean;
    hasCachedRemux?: boolean;
    // Phase 4 of docs/pts-based-frame-mapping.md: backend now returns
    // ffprobe's r_frame_rate / avg_frame_rate plus a VFR flag so the
    // editor can drive its frame-step UI off the real fps and the
    // browser can flag VFR captures (USB webcams in dim labs are
    // essentially always VFR).
    rFrameRate?: number | null;
    avgFrameRate?: number | null;
    isVfr?: boolean | null;
}

interface ProcessingJob {
    action: 'remux' | 'transcode';
    percent: number | null;
    processed: number | null;
    total: number | null;
    eventSource: EventSource;
}

interface IProps {
    videoDirectory: string;
    videoFiles: string[];
    videoCsvOverrides: Record<string, boolean>;
    currentFileName: string;
    currentCsvName: string;
    labelNames: LabelName[];
    isOpen: boolean;
    onToggleOpen: () => void;
    updateVideoDirectory: (dir: string) => any;
    updateVideoFiles: (files: string[]) => any;
    clearVideoHasCsv: (filename: string) => any;
    addImageData: (imageData: ImageData[]) => any;
    updateActiveImageIndex: (index: number) => any;
    updateImageData: (imageData: ImageData[]) => any;
    updateLabelNames: (labels: LabelName[]) => any;
    updateSubjects: (subjects: Subject[]) => any;
    updateActivePopupType: (popupType: PopupWindowType) => any;
    updatePopupPayload: (payload: unknown) => any;
    submitNewNotification: typeof submitNewNotification;
}

interface RenamingCsv {
    file: string;          // owning video filename
    csvName: string;       // the CSV being renamed
    variant: string;       // editable portion AFTER the locked `{base}_` prefix; never carries its own underscore
    error: string | null;
}

interface DeleteCsvTarget {
    file: string;
    csvName: string;
}

const formatDurationShort = (seconds: number | null | undefined): string => {
    if (seconds == null || !Number.isFinite(seconds)) return '—:—';
    const s = Math.max(0, Math.floor(seconds));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
    return `${m}:${String(ss).padStart(2, '0')}`;
};

const codecLabel = (codec: string | undefined): string =>
    codec ? codec.toUpperCase().replace('H264', 'H.264').replace('H265', 'H.265') : 'unknown';

const containerLabel = (container: string | undefined): string => {
    if (!container) return 'unknown';
    if (container === 'matroska') return 'Matroska';
    return container.toUpperCase();
};

const FileBrowser: React.FC<IProps> = (props) => {
    const [directory, setDirectory] = useState(props.videoDirectory || '');
    const [files, setFiles] = useState<string[]>(props.videoFiles || []);
    const [filesInfo, setFilesInfo] = useState<FileInfo[]>([]);
    const [loading, setLoading] = useState(false);
    const [expandedFile, setExpandedFile] = useState<string | null>(null);
    const [processingJobs, setProcessingJobs] = useState<Map<string, ProcessingJob>>(new Map());
    // Which CSV is currently loaded for a given video. Lets the user pick
    // among `{base}.csv`, `{base}_v2.csv`, etc. without losing the choice
    // when re-rendering. Keyed by the video filename.
    const [activeCsvByFile, setActiveCsvByFile] = useState<Map<string, string>>(new Map());
    // Inline-rename UI state for the per-video CSV rows.
    const [renaming, setRenaming] = useState<RenamingCsv | null>(null);
    // Two-step CSV deletion: the first click arms the row, the second commits.
    const [deleteTarget, setDeleteTarget] = useState<DeleteCsvTarget | null>(null);
    const [deletingCsvKey, setDeletingCsvKey] = useState<string | null>(null);

    // Auto-save in Editor.tsx flips `videoCsvOverrides[file]` to true the
    // first time a video's labels are persisted, before the next directory
    // rescan picks the new CSV up. Use it to mask the stale `info.hasCsv`
    // so the sidebar and click-to-load both react immediately.
    const hasCsvFor = (file: string, info: FileInfo | undefined): boolean =>
        Boolean(info?.hasCsv || props.videoCsvOverrides[file]);

    useEffect(() => {
        if (props.videoDirectory) {
            setDirectory(props.videoDirectory);
            fetchFilesInfo(props.videoDirectory);
        }
        if (props.videoFiles?.length) setFiles(props.videoFiles);
    }, []);

    // Whenever the active file changes (e.g. user picked a leaf), auto-expand
    // it so the freshly-loaded media is visible in context.
    useEffect(() => {
        if (props.currentFileName && files.includes(props.currentFileName)) {
            setExpandedFile(props.currentFileName);
        }
    }, [props.currentFileName]);

    // Tear down any in-flight SSE connections on unmount so we don't keep
    // ffmpeg processes pinned to a closed page.
    useEffect(() => {
        return () => {
            processingJobs.forEach(job => job.eventSource.close());
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const fetchFilesInfo = async (dir: string) => {
        const cleanDir = dir.replace(/\/+$/, '') || '/';
        try {
            const res = await fetch(`${API_URL}/api/files?dir=${encodeURIComponent(cleanDir)}`);
            if (res.ok) {
                const data = await res.json();
                setFilesInfo(data.filesInfo || []);
            }
        } catch {}
    };

    const loadFiles = async () => {
        if (!directory) return;
        setLoading(true);
        const cleanDir = directory.replace(/\/+$/, '') || '/';
        try {
            const res = await fetch(`${API_URL}/api/files?dir=${encodeURIComponent(cleanDir)}`);
            if (res.ok) {
                const data = await res.json();
                const newFiles = data.files || [];
                setFiles(newFiles);
                setFilesInfo(data.filesInfo || []);
                props.updateVideoDirectory(cleanDir);
                props.updateVideoFiles(newFiles);
            }
        } catch (err) {
            console.error('Failed to fetch files', err);
        }
        setLoading(false);
    };

    const toggleExpand = (file: string) => {
        setExpandedFile(prev => prev === file ? null : file);
    };

    const startProcess = (file: string, action: 'remux' | 'transcode') => {
        if (processingJobs.has(file)) return;

        const cleanDir = directory.replace(/\/+$/, '') || '/';
        const url = `${API_URL}/api/files/${encodeURIComponent(file)}/process-stream?dir=${encodeURIComponent(cleanDir)}&action=${action}`;
        const es = new EventSource(url);

        setProcessingJobs(prev => {
            const next = new Map(prev);
            next.set(file, { action, percent: null, processed: null, total: null, eventSource: es });
            return next;
        });

        const finish = (success: boolean, msg?: string) => {
            es.close();
            setProcessingJobs(prev => {
                const next = new Map(prev);
                next.delete(file);
                return next;
            });
            if (success) {
                setFilesInfo(prev => prev.map(fi =>
                    fi.name === file
                        ? { ...fi, ...(action === 'remux' ? { hasCachedRemux: true } : { hasCachedH264: true }) }
                        : fi
                ));
                props.submitNewNotification(NotificationUtil.createMessageNotification({
                    header: action === 'remux' ? 'Repack complete' : 'Re-encode complete',
                    description: `"${file}" is ready to play in the browser.`,
                }));
            } else if (msg) {
                props.submitNewNotification(NotificationUtil.createMessageNotification({
                    header: action === 'remux' ? 'Repack failed' : 'Re-encode failed',
                    description: msg,
                }));
            }
        };

        es.onmessage = (ev) => {
            try {
                const data = JSON.parse(ev.data);
                if (data.type === 'progress') {
                    setProcessingJobs(prev => {
                        const next = new Map(prev);
                        const existing = next.get(file);
                        if (existing) {
                            next.set(file, {
                                ...existing,
                                percent: data.percent,
                                processed: data.processed,
                                total: data.total,
                            });
                        }
                        return next;
                    });
                } else if (data.type === 'complete' || data.type === 'noop') {
                    finish(true);
                } else if (data.type === 'error') {
                    finish(false, data.message || 'ffmpeg error');
                }
            } catch {}
        };

        es.onerror = () => {
            // EventSource auto-reconnects on transient errors; close to be safe.
            // If we already received `complete`, finish() already closed it.
            if (es.readyState !== EventSource.CLOSED) {
                finish(false, 'Connection lost');
            }
        };
    };

    const cancelProcess = (file: string) => {
        const job = processingJobs.get(file);
        if (!job) return;
        job.eventSource.close();
        setProcessingJobs(prev => {
            const next = new Map(prev);
            next.delete(file);
            return next;
        });
    };

    // A CSV becomes active only after the user explicitly clicks a CSV row or
    // creates a new one. Clicking the media leaf should load the video alone,
    // without silently choosing the canonical `{base}.csv`.
    const activeCsvNameFor = (
        file: string,
        info: FileInfo | undefined,
    ): string | undefined => {
        const remembered = activeCsvByFile.get(file);
        if (remembered && (info?.csvFiles?.includes(remembered) ?? true)) return remembered;
        return undefined;
    };

    const playFile = async (file: string, opts?: { csvName?: string }) => {
        setDeleteTarget(null);
        const info = filesInfo.find(fi => fi.name === file);
        const cleanDir = directory.replace(/\/+$/, '') || '/';
        const action: ServeAction = info ? decideServeAction(info) : 'transcode';

        // The expansion panel itself is the place to start a process for files
        // that need one — only auto-fetch when the action is non-blocking.
        if (action === 'transcode' && !info?.hasCachedH264) return;

        const videoUrl = `${API_URL}/api/files/${file}?dir=${encodeURIComponent(cleanDir)}&action=${action}`;
        // Prefer avg_frame_rate (= n_frames / duration) over r_frame_rate
        // (= nominal capture rate). On a clean CFR file the two agree;
        // on VFR they don't and avg is the right thing for frame-step
        // UI math (`time = frame / fps`).
        const fr = info?.avgFrameRate ?? info?.rFrameRate ?? undefined;
        const newImage = ImageDataUtil.createImageDataFromUrl(
            videoUrl, file, fr ?? undefined,
        );
        (newImage as any).videoPath = cleanDir;
        const csvName = opts?.csvName;
        if (csvName) {
            // Stamp the active CSV onto imageData so Editor.tsx writes back to
            // the same file the user picked (instead of always `{base}.csv`).
            (newImage as any).csvName = csvName;
        }
        props.updateImageData([newImage]);
        props.updateActiveImageIndex(0);
        setActiveCsvByFile(prev => {
            const next = new Map(prev);
            if (csvName) next.set(file, csvName);
            else next.delete(file);
            return next;
        });

        // Media clicks stop here: the user chooses the annotation CSV from the
        // Annotations section below.
        if (!csvName) return;

        const canonicalCsv = getFileBase(file) + '.csv';
        const hasListedCsv = info?.csvFiles?.includes(csvName) ?? false;
        const hasLegacyCanonicalCsv = !info?.csvFiles?.length && csvName === canonicalCsv && hasCsvFor(file, info);
        const csvExists = hasListedCsv || hasLegacyCanonicalCsv;
        if (csvExists) {
            try {
                const res = await fetch(
                    `${API_URL}/api/files/${file}/csv?dir=${encodeURIComponent(cleanDir)}&csvName=${encodeURIComponent(csvName)}`
                );
                if (res.ok) {
                    const csvText = await res.text();
                    const { rows, meta } = CSVImporter.parseCSVWithMeta(csvText);
                    const importer = new CSVImporter([]);
                    const { imagesData: updatedImages, labelNames, subjects } = importer.applyLabels([newImage], rows, meta);
                    props.updateLabelNames(labelNames);
                    if (subjects && subjects.length > 0) {
                        props.updateSubjects(subjects);
                    }
                    props.updateImageData(updatedImages);
                    const firstMissing = labelNames.find(l => !l.shortcut);
                    if (firstMissing) {
                        setPendingFocusLabelId(firstMissing.id, 'shortcut');
                        props.updateActivePopupType(PopupWindowType.UPDATE_LABEL);
                    }
                }
            } catch {}
        }
        // Empty/new CSV: leave the editor in its empty annotation state. The
        // explicit "New" path opens the behavior-types modal.
    };

    // Pick a fresh CSV name for "Save As": `{base}_2.csv`, `{base}_3.csv`, …
    // skipping any that already exist on disk.
    const newCsvName = (file: string, info: FileInfo | undefined): string => {
        const base = getFileBase(file);
        const taken = new Set(info?.csvFiles ?? []);
        for (let i = 2; i < 1000; i++) {
            const candidate = `${base}_${i}.csv`;
            if (!taken.has(candidate)) return candidate;
        }
        // Pathological fallback — should never trigger in practice.
        return `${base}_${Date.now()}.csv`;
    };

    const startNewCsv = (file: string) => {
        setDeleteTarget(null);
        const info = filesInfo.find(fi => fi.name === file);
        const csvName = newCsvName(file, info);
        // Switch immediately to the new (still-empty) CSV. The editor's
        // auto-save will materialise the file on disk on the first edit.
        playFile(file, { csvName });
        // The explicit "create annotation file" path is the one and only
        // trigger for the behavior-types modal now that click-to-play no
        // longer auto-opens it. Without this, a brand-new CSV would be
        // created with no behaviors defined and the user wouldn't know to
        // configure them.
        props.updateActivePopupType(PopupWindowType.INSERT_LABEL_NAMES);
        // Make the new CSV visible in the sidebar without waiting for a
        // directory rescan. We treat this as an optimistic update.
        setFilesInfo(prev => prev.map(fi =>
            fi.name === file
                ? { ...fi, csvFiles: [...(fi.csvFiles ?? []), csvName].sort() }
                : fi
        ));
    };

    const getFileExt = (filename: string) => {
        const dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(dot) : '';
    };

    const getFileBase = (filename: string) => {
        const dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    };

    // Open the read-only annotation table preview for a CSV.
    const openPreview = (file: string, csvName: string, dir: string) => {
        setDeleteTarget(null);
        props.updatePopupPayload({ file, csvName, dir });
        props.updateActivePopupType(PopupWindowType.ANNOTATION_PREVIEW);
    };

    // Pull a CSV down to the user's machine via the same endpoint the
    // preview popup uses. We fetch through JS (rather than a plain <a
    // download>) so the API_URL prefix and any future auth headers stay
    // centralised, and so we can surface load failures as notifications.
    const downloadCsv = async (file: string, csvName: string, dir: string) => {
        setDeleteTarget(null);
        try {
            const params = new URLSearchParams({ dir, csvName });
            const res = await fetch(
                `${API_URL}/api/files/${encodeURIComponent(file)}/csv?${params}`,
            );
            if (!res.ok) {
                props.submitNewNotification(NotificationUtil.createMessageNotification({
                    header: 'Download failed',
                    description: res.status === 404
                        ? `${csvName} was not found on the server.`
                        : `Server returned ${res.status}`,
                }));
                return;
            }
            const text = await res.text();
            const blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
            saveAs(blob, csvName);
        } catch {
            props.submitNewNotification(NotificationUtil.createMessageNotification({
                header: 'Download failed',
                description: 'Network error while downloading CSV',
            }));
        }
    };

    // Extract the variant identifier from a CSV name. CSV names follow the
    // `{videoBase}_{variant}.csv` convention for variants, or `{videoBase}.csv`
    // for the canonical file. The rename UI exposes only the variant — the
    // `{videoBase}_` prefix and `.csv` extension are locked text — so we
    // pre-fill the input with the slice between them. Returns '' for the
    // canonical file or any name that doesn't follow the convention; the
    // input starts empty in those cases and the user types a fresh variant.
    const extractVariant = (csvName: string, videoBase: string): string => {
        const stem = csvName.replace(/\.csv$/i, '');
        const prefix = `${videoBase}_`;
        return stem.startsWith(prefix) ? stem.slice(prefix.length) : '';
    };

    const beginRename = (file: string, csvName: string, videoBase: string) => {
        setDeleteTarget(null);
        setRenaming({ file, csvName, variant: extractVariant(csvName, videoBase), error: null });
    };

    const cancelRename = () => setRenaming(null);

    // Validate the variant identifier only — the `{videoBase}_` prefix
    // and `.csv` extension are fixed by the locked-text UI, so we never
    // see a malformed full name here. Mirrors `_resolve_csv_name`
    // server-side: reject path separators, reject a redundant leading
    // underscore (the prefix already supplies it), and detect collisions
    // with sibling CSVs. Empty variants are tolerated by the validator
    // and gated by `canSave` instead so the user doesn't see a scary red
    // error every time they backspace through the input.
    const validateRenameVariant = (
        variant: string,
        currentName: string,
        siblings: string[],
        videoBase: string,
    ): string | null => {
        if (variant.includes('/') || variant.includes('\\')) return 'No slashes allowed';
        if (variant.startsWith('_')) return 'No leading underscore — it is added for you';
        if (!variant) return null;
        const fullName = `${videoBase}_${variant}.csv`;
        if (fullName === currentName) return null;
        if (siblings.includes(fullName)) return 'A CSV with that name already exists';
        return null;
    };

    const commitRename = async (
        file: string,
        oldName: string,
        variant: string,
        videoBase: string,
        dir: string,
    ) => {
        const newName = `${videoBase}_${variant}.csv`;
        if (newName === oldName) {
            cancelRename();
            return;
        }
        try {
            const params = new URLSearchParams({ dir, oldName, newName });
            const res = await fetch(`${API_URL}/api/files/${encodeURIComponent(file)}/rename-csv?${params}`, {
                method: 'POST',
            });
            if (!res.ok) {
                let detail = 'Rename failed';
                try { const data = await res.json(); detail = data.detail || detail; } catch {}
                setRenaming(prev => prev ? { ...prev, error: detail } : prev);
                return;
            }
            // Mirror the rename through every piece of local state that
            // remembered the old name. Otherwise the next click on this
            // row would 404, and the editor would still POST saves to
            // the old name (since `imageData.csvName` is sticky).
            setFilesInfo(prev => prev.map(fi => fi.name === file
                ? { ...fi, csvFiles: (fi.csvFiles ?? []).map(n => n === oldName ? newName : n).sort() }
                : fi
            ));
            setActiveCsvByFile(prev => {
                const next = new Map(prev);
                if (next.get(file) === oldName) next.set(file, newName);
                return next;
            });
            cancelRename();
            props.submitNewNotification(NotificationUtil.createMessageNotification({
                header: 'Renamed',
                description: `${oldName} → ${newName}`,
            }));
        } catch (e: any) {
            setRenaming(prev => prev ? { ...prev, error: 'Rename failed (network error)' } : prev);
        }
    };

    const csvActionKey = (file: string, csvName: string): string => `${file}::${csvName}`;

    const beginDeleteCsv = (file: string, csvName: string) => {
        setRenaming(null);
        setDeleteTarget({ file, csvName });
    };

    const cancelDeleteCsv = () => setDeleteTarget(null);

    const commitDeleteCsv = async (
        file: string,
        csvName: string,
        dir: string,
        siblings: string[],
    ) => {
        const key = csvActionKey(file, csvName);
        const nextCsvs = siblings.filter(name => name !== csvName);
        setDeletingCsvKey(key);

        try {
            const params = new URLSearchParams({ dir, csvName });
            const res = await fetch(`${API_URL}/api/files/${encodeURIComponent(file)}/csv?${params}`, {
                method: 'DELETE',
            });

            if (!res.ok && res.status !== 404) {
                let detail = 'Delete failed';
                try { const data = await res.json(); detail = data.detail || detail; } catch {}
                props.submitNewNotification(NotificationUtil.createMessageNotification({
                    header: 'Delete failed',
                    description: detail,
                }));
                return;
            }

            setFilesInfo(prev => prev.map(fi => fi.name === file
                ? { ...fi, hasCsv: nextCsvs.length > 0, csvFiles: nextCsvs }
                : fi
            ));
            if (nextCsvs.length === 0) {
                props.clearVideoHasCsv(file);
            }
            setActiveCsvByFile(prev => {
                const next = new Map(prev);
                if (next.get(file) === csvName) {
                    next.delete(file);
                }
                return next;
            });
            setDeleteTarget(null);

            if (props.currentFileName === file && props.currentCsvName === csvName) {
                void playFile(file);
            }

            props.submitNewNotification(NotificationUtil.createMessageNotification({
                header: 'Deleted',
                description: res.status === 404
                    ? `${csvName} was already missing; removed it from the list.`
                    : `${csvName} was deleted.`,
            }));
        } catch {
            props.submitNewNotification(NotificationUtil.createMessageNotification({
                header: 'Delete failed',
                description: 'Network error while deleting CSV',
            }));
        } finally {
            setDeletingCsvKey(prev => prev === key ? null : prev);
        }
    };

    interface CsvRowOptions {
        ownerFile: string;
        csvName: string;
        dir: string;
        siblings: string[];   // existing CSVs for this video — for collision detection
        videoBase: string;
        isActive: boolean;
        onLoad: () => void;
    }

    const renderCsvRow = (opts: CsvRowOptions) => {
        const { ownerFile, csvName, dir, siblings, videoBase, isActive, onLoad } = opts;
        const isRenaming = renaming?.file === ownerFile && renaming.csvName === csvName;
        const isConfirmingDelete = deleteTarget?.file === ownerFile && deleteTarget.csvName === csvName;
        const isDeleting = deletingCsvKey === csvActionKey(ownerFile, csvName);
        const variantError = isRenaming
            ? validateRenameVariant(renaming!.variant, csvName, siblings.filter(s => s !== csvName), videoBase)
            : null;
        const proposed = isRenaming ? `${videoBase}_${renaming!.variant}.csv` : csvName;
        const canSave = isRenaming && !variantError && renaming!.variant.length > 0 && proposed !== csvName;
        return (
            <div className={classNames('CsvRow', { active: isActive, renaming: isRenaming, deleting: isConfirmingDelete })} key={csvName}>
                <button
                    type="button"
                    className="CsvRowMain"
                    onClick={(e) => { e.stopPropagation(); if (!isRenaming && !isConfirmingDelete) onLoad(); }}
                    disabled={isRenaming || isConfirmingDelete}
                >
                    <span className="LeafIcon" aria-hidden="true">
                        <svg width="10" height="10" viewBox="0 0 10 10">
                            <rect x="1.5" y="1" width="7" height="8" rx="1" stroke="currentColor" strokeWidth="1" fill="none"/>
                            <path d="M3 4 H7 M3 6 H7" stroke="currentColor" strokeWidth="0.8"/>
                        </svg>
                    </span>
                    {isRenaming ? (
                        <span className={classNames('RenameField', { invalid: !!variantError })}>
                            <span className="LockedPrefix" title="Locked: video base name and underscore are fixed">{videoBase}_</span>
                            <input
                                className="RenameSuffix"
                                value={renaming!.variant}
                                placeholder="variant"
                                aria-label="Variant identifier"
                                onChange={(e) => setRenaming(prev => prev ? { ...prev, variant: e.target.value, error: null } : prev)}
                                onClick={(e) => e.stopPropagation()}
                                onKeyDown={(e) => {
                                    e.stopPropagation();
                                    if (e.key === 'Enter' && canSave) {
                                        void commitRename(ownerFile, csvName, renaming!.variant, videoBase, dir);
                                    } else if (e.key === 'Escape') {
                                        cancelRename();
                                    }
                                }}
                                autoFocus
                                spellCheck={false}
                            />
                            <span className="LockedExt">.csv</span>
                        </span>
                    ) : (
                        <Tooltip text={csvName}>
                            <span className="LeafName">{csvName}</span>
                        </Tooltip>
                    )}
                    {!isRenaming && <span className="LeafTag derived">labels</span>}
                </button>
                <div className="CsvRowActions" onClick={(e) => e.stopPropagation()}>
                    {isRenaming ? (
                        <>
                            <button
                                type="button"
                                className="RowAction confirm"
                                disabled={!canSave}
                                onClick={() => void commitRename(ownerFile, csvName, renaming!.variant, videoBase, dir)}
                                title="Save name"
                            >
                                <svg width="17" height="17" viewBox="0 0 12 12" fill="none">
                                    <path d="M2 6.5l2.6 2.5L10 3.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                            </button>
                            <button
                                type="button"
                                className="RowAction cancel"
                                onClick={cancelRename}
                                title="Discard"
                            >
                                <svg width="17" height="17" viewBox="0 0 12 12" fill="none">
                                    <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
                                </svg>
                            </button>
                        </>
                    ) : isConfirmingDelete ? (
                        <>
                            <button
                                type="button"
                                className="RowAction delete confirmDelete"
                                disabled={isDeleting}
                                onClick={() => void commitDeleteCsv(ownerFile, csvName, dir, siblings)}
                                title="Delete CSV"
                                aria-label={`Delete ${csvName}`}
                            >
                                <svg width="17" height="17" viewBox="0 0 14 14" fill="none">
                                    <path d="M4 5v6M7 5v6M10 5v6" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round"/>
                                    <path d="M2.5 3.5h9M5.2 3.5l.5-1h2.6l.5 1M3.5 3.5l.5 9h6l.5-9" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                            </button>
                            <button
                                type="button"
                                className="RowAction cancel"
                                disabled={isDeleting}
                                onClick={cancelDeleteCsv}
                                title="Cancel delete"
                                aria-label="Cancel delete"
                            >
                                <svg width="17" height="17" viewBox="0 0 12 12" fill="none">
                                    <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
                                </svg>
                            </button>
                        </>
                    ) : (
                        <>
                            <button
                                type="button"
                                className="RowAction preview"
                                onClick={() => openPreview(ownerFile, csvName, dir)}
                                title="Preview annotations"
                            >
                                <svg width="17" height="17" viewBox="0 0 16 16" fill="none">
                                    <path d="M1.5 8s2.5-4.5 6.5-4.5S14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z" stroke="currentColor" strokeWidth="1.1" fill="none"/>
                                    <circle cx="8" cy="8" r="1.6" stroke="currentColor" strokeWidth="1.1" fill="none"/>
                                </svg>
                            </button>
                            <button
                                type="button"
                                className="RowAction download"
                                onClick={() => void downloadCsv(ownerFile, csvName, dir)}
                                title="Download CSV"
                                aria-label={`Download ${csvName}`}
                            >
                                <svg width="17" height="17" viewBox="0 0 16 16" fill="none">
                                    <path d="M8 2v8m0 0l-3-3m3 3l3-3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
                                    <path d="M3 12v1.5A0.5 0.5 0 0 0 3.5 14h9a0.5 0.5 0 0 0 0.5-0.5V12" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                            </button>
                            <button
                                type="button"
                                className="RowAction rename"
                                onClick={() => beginRename(ownerFile, csvName, videoBase)}
                                title="Rename"
                            >
                                <svg width="17" height="17" viewBox="0 0 14 14" fill="none">
                                    <path d="M2 12h2.2L11 5.2 8.8 3 2 9.8V12z" stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round" fill="none"/>
                                    <path d="M8.4 3.4l2.2 2.2" stroke="currentColor" strokeWidth="1.1"/>
                                </svg>
                            </button>
                            <button
                                type="button"
                                className="RowAction delete"
                                onClick={() => beginDeleteCsv(ownerFile, csvName)}
                                title="Delete CSV"
                                aria-label={`Delete ${csvName}`}
                            >
                                <svg width="17" height="17" viewBox="0 0 14 14" fill="none">
                                    <path d="M4 5v6M7 5v6M10 5v6" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round"/>
                                    <path d="M2.5 3.5h9M5.2 3.5l.5-1h2.6l.5 1M3.5 3.5l.5 9h6l.5-9" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                            </button>
                        </>
                    )}
                </div>
                {isRenaming && (renaming!.error || variantError) && (
                    <div className="RenameError">{renaming!.error || variantError}</div>
                )}
                {isConfirmingDelete && (
                    <div className="DeleteWarning">
                        {isDeleting ? `Deleting ${csvName}...` : `Delete ${csvName} from disk?`}
                    </div>
                )}
            </div>
        );
    };

    if (!props.isOpen) {
        return (
            <div
                className="FileBrowser collapsed"
                onClick={props.onToggleOpen}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        props.onToggleOpen();
                    }
                }}
                aria-label="Expand file browser"
            >
                <span className="ToggleIcon">&raquo;</span>
            </div>
        );
    }

    const renderHeaderBadge = (info: FileInfo | undefined, job: ProcessingJob | undefined, action: ServeAction | null) => {
        if (job) {
            const pct = job.percent != null ? Math.round(job.percent) : null;
            const verb = job.action === 'remux' ? 'Wrapping' : 'Re-encoding';
            return (
                <span className="StatusPill working" title={`${verb} — ${pct ?? '—'}%`}>
                    <span className="StatusPillSpinner" />
                    {pct != null ? `${pct}%` : '…'}
                </span>
            );
        }
        if (info?.hasCachedH264 || info?.hasCachedRemux || action === 'raw') {
            return <span className="StatusPill ready" title="This file plays in your browser">Ready</span>;
        }
        if (action === 'remux') {
            return <span className="StatusPill wrap" title="Container needs an MP4 wrapper to play in the browser">Wrap</span>;
        }
        if (action === 'transcode') {
            return <span className="StatusPill convert" title="Codec must be re-encoded to play in the browser">Re-encode</span>;
        }
        return <span className="StatusPill unknown">…</span>;
    };

    return (
        <div className={classNames('FileBrowser', { empty: files.length === 0 })}>
            <div className="BrowserHeader">
                <div className="HeaderCopy">
                    <span className="HeaderLabel">Files</span>
                    <span className="HeaderSubtext">Drag edge to resize</span>
                </div>
                <span className="FileCount">{files.length}</span>
                <button type='button' className="CollapseBtn" onClick={props.onToggleOpen} aria-label="Collapse file browser">
                    &laquo;
                </button>
            </div>
            <div className="DirSection">
                <PathPicker
                    value={directory}
                    onChange={setDirectory}
                    onSubmit={loadFiles}
                    placeholder="/path/to/videos"
                    storageKey="editor-files"
                    previewExtensions=".mp4,.avi,.mov,.mkv,.webm,.csv"
                />
                <button
                    type='button'
                    className="LoadFolderBtn"
                    onClick={loadFiles}
                    disabled={!directory}
                    aria-label="Load folder"
                >
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                        <path d="M8 3v7m0 0l-3-3m3 3l3-3M3 13h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    Load folder
                </button>
            </div>
            <div className="FileList">
                {loading && <div className="LoadingIndicator">Loading...</div>}
                {!loading && files.length === 0 && (
                    <div className="EmptyState">
                        <div className="EmptyStateTitle">No files loaded</div>
                        <div className="EmptyStateText">Choose a server folder to populate this browser.</div>
                    </div>
                )}
                {!loading && files.map(file => {
                    const info = filesInfo.find(fi => fi.name === file);
                    const action: ServeAction | null = info ? decideServeAction(info) : null;
                    const job = processingJobs.get(file);
                    const expanded = expandedFile === file;
                    const isActive = file === props.currentFileName;
                    const sourceNativelyPlayable = info ? browserCanPlay(info.codec, info.container) : false;
                    const cacheName = info?.hasCachedH264
                        ? `${file}.h264.mp4`
                        : info?.hasCachedRemux ? `${file}.remux.mp4` : null;
                    const cacheKind: 'h264' | 'remux' | null = info?.hasCachedH264 ? 'h264' : info?.hasCachedRemux ? 'remux' : null;

                    return (
                        <div
                            key={file}
                            className={classNames('FileGroup', { active: isActive, expanded })}
                        >
                            <button
                                type="button"
                                className="FileRow"
                                onClick={() => toggleExpand(file)}
                                aria-expanded={expanded}
                            >
                                <span className="FileMain">
                                    <span className="FileBase">{getFileBase(file)}</span>
                                    <span className="FileExt">{getFileExt(file)}</span>
                                </span>
                                <span className="FileMeta">
                                    {renderHeaderBadge(info, job, action)}
                                    <span className={classNames('Chevron', { open: expanded })} aria-hidden="true">
                                        <svg width="10" height="6" viewBox="0 0 10 6" fill="none">
                                            <path d="M1 1L5 5L9 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                                        </svg>
                                    </span>
                                </span>
                            </button>

                            <div className="FileExpansion" data-expanded={expanded}>
                                <div className="FileExpansionInner">
                                    {/* Compatibility row — codec/container with status */}
                                    <div className="ExpRow ExpCompat">
                                        <span className="ExpRowLabel">Format</span>
                                        <span className="ExpRowValue">
                                            <span className="CodecChip">{codecLabel(info?.codec)}</span>
                                            <span className="ChipSep">in</span>
                                            <span className="ContainerChip">{containerLabel(info?.container)}</span>
                                            {info?.avgFrameRate ? (
                                                <span className="FpsChip" title={
                                                    info?.rFrameRate && info?.avgFrameRate &&
                                                    Math.abs(info.rFrameRate - info.avgFrameRate) > 0.01
                                                        ? `nominal ${info.rFrameRate.toFixed(2)} / actual ${info.avgFrameRate.toFixed(2)}`
                                                        : undefined
                                                }>
                                                    {info.avgFrameRate.toFixed(info.avgFrameRate % 1 < 0.01 ? 0 : 2)} fps
                                                </span>
                                            ) : null}
                                            {info?.isVfr === true && (
                                                <span
                                                    className="VfrBadge"
                                                    title="Variable frame rate — frame timing is irregular. TRACE handles this correctly via per-frame PTS lookup, no re-encoding."
                                                >
                                                    VFR
                                                </span>
                                            )}
                                        </span>
                                    </div>
                                    <div className="ExpRow ExpCompatNote">
                                        {action === 'raw' && !cacheName && (
                                            <span className="CompatNote ok">Plays in your browser as-is.</span>
                                        )}
                                        {action === 'remux' && !cacheName && (
                                            <span className="CompatNote warn">Not browser-compatible. The codec is fine — only the container needs to be repacked into MP4. This is fast and lossless.</span>
                                        )}
                                        {action === 'transcode' && !cacheName && (
                                            <span className="CompatNote warn">Not browser-compatible. The codec must be re-encoded to H.264. This takes longer and is a quality trade-off.</span>
                                        )}
                                        {cacheName && (
                                            <span className="CompatNote ok">A browser-compatible copy is ready. The original stays untouched on disk.</span>
                                        )}
                                    </div>

                                    {/* Media tree */}
                                    <div className="ExpSection">
                                        <div className="ExpSectionLabel">Media</div>

                                        {/* Source leaf */}
                                        <Tooltip text={sourceNativelyPlayable
                                            ? `${file} — original; your browser can play this directly`
                                            : `${file} — original; your browser cannot play this file`}
                                        >
                                            <button
                                                type="button"
                                                className={classNames('Leaf', { disabled: !sourceNativelyPlayable, active: isActive && !cacheKind })}
                                                disabled={!sourceNativelyPlayable}
                                                onClick={(e) => { e.stopPropagation(); if (sourceNativelyPlayable) playFile(file); }}
                                            >
                                                <span className="LeafIcon" aria-hidden="true">
                                                    {sourceNativelyPlayable ? (
                                                        <svg width="10" height="10" viewBox="0 0 10 10"><path d="M2 1.5 L8.5 5 L2 8.5 Z" fill="currentColor"/></svg>
                                                    ) : (
                                                        <svg width="10" height="10" viewBox="0 0 10 10"><circle cx="5" cy="5" r="3.5" stroke="currentColor" strokeWidth="1" fill="none"/></svg>
                                                    )}
                                                </span>
                                                <span className="LeafName">{file}</span>
                                                {sourceNativelyPlayable
                                                    ? <span className="LeafTag compatible">Browser ✓</span>
                                                    : <span className="LeafTag incompatible">Not compatible</span>}
                                            </button>
                                        </Tooltip>

                                        {/* Cached derivative leaf, when present */}
                                        {cacheName && (
                                            <Tooltip text={cacheKind === 'h264'
                                                ? `${cacheName} — re-encoded copy; plays in your browser`
                                                : `${cacheName} — repacked copy; plays in your browser`}
                                            >
                                                <button
                                                    type="button"
                                                    className={classNames('Leaf', { active: isActive })}
                                                    onClick={(e) => { e.stopPropagation(); playFile(file); }}
                                                >
                                                    <span className="LeafIcon" aria-hidden="true">
                                                        <svg width="10" height="10" viewBox="0 0 10 10"><path d="M2 1.5 L8.5 5 L2 8.5 Z" fill="currentColor"/></svg>
                                                    </span>
                                                    <span className="LeafName">{cacheName}</span>
                                                    <span className="LeafTag compatible">Browser ✓</span>
                                                </button>
                                            </Tooltip>
                                        )}

                                        {/* Action: process the source, when no cache exists yet */}
                                        {!cacheKind && action === 'remux' && !job && (
                                            <button
                                                type="button"
                                                className="ActionRow remux"
                                                onClick={(e) => { e.stopPropagation(); startProcess(file, 'remux'); }}
                                                title="Stream-copy the video into an MP4 container — no quality loss"
                                            >
                                                <span className="ActionIcon" aria-hidden="true">
                                                    <svg width="11" height="11" viewBox="0 0 16 16" fill="none">
                                                        <path d="M3 5h10M3 8h10M3 11h6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
                                                    </svg>
                                                </span>
                                                <span className="ActionLabel">Make compatible</span>
                                                <span className="ActionHint">fast · lossless</span>
                                            </button>
                                        )}
                                        {!cacheKind && action === 'transcode' && !job && (
                                            <button
                                                type="button"
                                                className="ActionRow transcode"
                                                onClick={(e) => { e.stopPropagation(); startProcess(file, 'transcode'); }}
                                                title="Re-encode the video into H.264 — slower, with a small quality trade-off"
                                            >
                                                <span className="ActionIcon" aria-hidden="true">
                                                    <svg width="11" height="11" viewBox="0 0 16 16" fill="none">
                                                        <path d="M4 6a4 4 0 0 1 7-2.7M12 10a4 4 0 0 1-7 2.7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
                                                        <path d="M11 1.5v3h-3M5 14.5v-3h3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
                                                    </svg>
                                                </span>
                                                <span className="ActionLabel">Make compatible</span>
                                                <span className="ActionHint">slower · re-encode</span>
                                            </button>
                                        )}

                                        {/* Live progress, when a job is running */}
                                        {job && (
                                            <div className={classNames('ProgressRow', job.action)}>
                                                <div className="ProgressTopline">
                                                    <span className="ProgressVerb">{job.action === 'remux' ? 'Repacking' : 'Re-encoding'}…</span>
                                                    <span className="ProgressPct">{job.percent != null ? `${job.percent.toFixed(1)}%` : '—'}</span>
                                                </div>
                                                <div className="ProgressTrack">
                                                    <div
                                                        className={classNames('ProgressFill', { indeterminate: job.percent == null })}
                                                        style={job.percent != null ? { width: `${Math.min(100, Math.max(0, job.percent))}%` } : undefined}
                                                    />
                                                </div>
                                                <div className="ProgressBottomline">
                                                    <span className="ProgressTime">
                                                        {formatDurationShort(job.processed)} / {formatDurationShort(job.total)}
                                                    </span>
                                                    <button
                                                        type="button"
                                                        className="ProgressCancel"
                                                        onClick={(e) => { e.stopPropagation(); cancelProcess(file); }}
                                                    >
                                                        Cancel
                                                    </button>
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    {/* Annotation CSV(s) — one row per labeling. The user can
                                        keep multiple variants per video (rater A vs rater B,
                                        draft vs final) and switch by clicking a row. */}
                                    <div className="ExpSection">
                                        <div className="ExpSectionLabel">
                                            <span>Annotations</span>
                                            <button
                                                type="button"
                                                className="ExpAddCsv"
                                                onClick={(e) => { e.stopPropagation(); startNewCsv(file); }}
                                                title="Start a new annotation file for this video"
                                            >
                                                <svg width="9" height="9" viewBox="0 0 11 11" fill="none">
                                                    <path d="M5.5 1.6v7.8M1.6 5.5h7.8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                                                </svg>
                                                <span>New</span>
                                            </button>
                                        </div>
                                        {(() => {
                                            const csvList = info?.csvFiles && info.csvFiles.length > 0
                                                ? info.csvFiles
                                                : (hasCsvFor(file, info) ? [getFileBase(file) + '.csv'] : []);
                                            if (csvList.length === 0) {
                                                return (
                                                    <div className="Leaf empty">
                                                        <span className="LeafIcon" aria-hidden="true">
                                                            <svg width="10" height="10" viewBox="0 0 10 10">
                                                                <rect x="1.5" y="1" width="7" height="8" rx="1" stroke="currentColor" strokeWidth="1" strokeDasharray="1.5 1.5" fill="none"/>
                                                            </svg>
                                                        </span>
                                                        <span className="LeafName muted" title="No CSV yet — create one before annotating">
                                                            No CSV yet — create one before annotating
                                                        </span>
                                                    </div>
                                                );
                                            }
                                            const activeCsv = activeCsvNameFor(file, info);
                                            const cleanDir = directory.replace(/\/+$/, '') || '/';
                                            return csvList.map((csvName) => renderCsvRow({
                                                ownerFile: file,
                                                csvName,
                                                dir: cleanDir,
                                                siblings: csvList,
                                                videoBase: getFileBase(file),
                                                isActive: isActive && csvName === activeCsv,
                                                onLoad: () => playFile(file, { csvName }),
                                            }));
                                        })()}
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

const mapStateToProps = (state: AppState) => ({
    videoDirectory: state.general.videoDirectory,
    videoFiles: state.general.videoFiles,
    videoCsvOverrides: state.general.videoCsvOverrides,
    currentFileName: state.labels.imagesData[state.labels.activeImageIndex]?.fileData?.name || '',
    currentCsvName: (state.labels.imagesData[state.labels.activeImageIndex] as any)?.csvName || '',
    labelNames: state.labels.labels
});

const mapDispatchToProps = {
    updateVideoDirectory,
    updateVideoFiles,
    addImageData,
    updateActiveImageIndex,
    updateImageData,
    updateLabelNames,
    updateSubjects,
    clearVideoHasCsv,
    updateActivePopupType,
    updatePopupPayload,
    submitNewNotification
};

export default connect(mapStateToProps, mapDispatchToProps)(FileBrowser);
